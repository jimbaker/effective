"""What Redis Streams promises a bridge, measured rather than assumed.

Six probes, one per assumption the bridge makes about delivery, each predicted before it
ran. Each asserts what it observed, so a later server
version that behaves differently fails here rather than in the bridge.

These are PROBES, not a pin on our own code: they measure a third party, and their value is that
the bridge's recovery contract can cite a measurement. `EFFECTIVE_REQUIRE_REDIS=1` turns the skip
into a refusal to start, which is what `just redis-probe` sets.
"""

import os
import uuid

import pytest

redis = pytest.importorskip("redis", reason="the `bridge` extra is not installed")

REQUIRED = os.environ.get("EFFECTIVE_REQUIRE_REDIS") == "1"
URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

GROUP = "bridge"
IDLE_MS = 0
"""Milliseconds an entry must be idle before another consumer may claim it. Zero because these
probes want the reclaim to happen now; a deployed bridge sets it above its own poll interval, so
a consumer that is merely slow is not raced by its replacement."""


@pytest.fixture
def client():
    """A client on an empty keyspace, or a skip that says why."""
    conn = redis.Redis.from_url(URL, decode_responses=True)
    try:
        conn.ping()
    except redis.exceptions.ConnectionError as exc:
        if REQUIRED:
            pytest.fail(f"EFFECTIVE_REQUIRE_REDIS=1 and {URL} is not answering: {exc}")
        pytest.skip(f"no redis at {URL} (`just redis-up`)")
    yield conn
    conn.close()


@pytest.fixture
def stream(client):
    """A stream name no other test uses, deleted afterwards, so probes do not read each other's
    entries even under `-p randomly` or `-n auto`."""
    name = f"probe:{uuid.uuid4()}"
    yield name
    client.delete(name)


def read_new(client, stream, consumer, count=10):
    """One `XREADGROUP` for entries never delivered to this group."""
    got = client.xreadgroup(GROUP, consumer, {stream: ">"}, count=count)
    return got[0][1] if got else []


def read_pending(client, stream, consumer, count=10):
    """One `XREADGROUP` for what THIS consumer already holds unacked: id `0`, not `>`."""
    got = client.xreadgroup(GROUP, consumer, {stream: "0"}, count=count)
    return got[0][1] if got else []


def test_R1_an_unacked_entry_is_reclaimable_by_another_consumer(client, stream):
    client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    sent = client.xadd(stream, {"payload": "a"})

    delivered = read_new(client, stream, "consumer-1")
    assert [i for i, _ in delivered] == [sent], "the first consumer did not get the entry"

    # Never acked. A second consumer takes it over, and gets the payload with it: a reclaim that
    # returned only the id would leave the replacement knowing an arrival happened and unable to
    # say what it was.
    _, claimed, _ = client.xautoclaim(stream, GROUP, "consumer-2", min_idle_time=IDLE_MS)
    assert claimed == [(sent, {"payload": "a"})]


def test_R2_an_acked_entry_is_not_handed_back(client, stream):
    client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    sent = client.xadd(stream, {"payload": "a"})

    read_new(client, stream, "consumer-1")
    assert client.xack(stream, GROUP, sent) == 1

    # The consumer restarts and asks for what it still holds. An acked entry is not it.
    assert read_pending(client, stream, "consumer-1") == []
    assert client.xpending(stream, GROUP)["pending"] == 0


def test_R3_a_groups_start_position_decides_what_it_can_ever_see(client, stream):
    before = client.xadd(stream, {"payload": "before"})
    client.xgroup_create(stream, "from-now", id="$")
    client.xgroup_create(stream, "from-start", id="0")
    after = client.xadd(stream, {"payload": "after"})

    now = client.xreadgroup("from-now", "c", {stream: ">"}, count=10)[0][1]
    start = client.xreadgroup("from-start", "c", {stream: ">"}, count=10)[0][1]

    assert [i for i, _ in now] == [after], "a group created at $ saw an earlier entry"
    assert [i for i, _ in start] == [before, after]


def test_R4_an_id_is_stable_and_increasing_across_reads(client, stream):
    client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    ids = [client.xadd(stream, {"n": str(n)}) for n in range(3)]

    def as_pair(entry_id: str) -> tuple[int, int]:
        ms, seq = entry_id.split("-")
        return int(ms), int(seq)

    assert [as_pair(i) for i in ids] == sorted(as_pair(i) for i in ids)
    assert len(set(ids)) == 3, "two entries shared an id"

    # The same ids survive a read, a reclaim, and a second read: the identity a dedup would key on
    # does not depend on how the entry was reached.
    assert [i for i, _ in read_new(client, stream, "consumer-1")] == ids
    _, claimed, _ = client.xautoclaim(stream, GROUP, "consumer-2", min_idle_time=IDLE_MS)
    assert [i for i, _ in claimed] == ids
    assert [i for i, _ in read_pending(client, stream, "consumer-2")] == ids


def test_R5_trimming_drops_an_entry_that_is_still_pending(client, stream):
    """The hazard: retention trims by length alone, so an entry a group still owes can go."""
    client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    first = client.xadd(stream, {"payload": "first"})
    read_new(client, stream, "consumer-1")
    assert client.xpending(stream, GROUP)["pending"] == 1, "the entry should be held unacked"

    # Retention passes while the entry is still owed to a consumer.
    client.xadd(stream, {"payload": "second"}, maxlen=1, approximate=False)
    assert [i for i, _ in client.xrange(stream)] != [first], "the trim did not drop it"

    # The pending list still names the id, so the bridge can tell an arrival is owed. What it
    # cannot do is deliver it: the payload is gone with the entry.
    _, claimed, deleted = client.xautoclaim(stream, GROUP, "consumer-2", min_idle_time=IDLE_MS)
    assert claimed == [], "a trimmed entry came back with a payload"
    assert deleted == [first], "the reclaim did not report the id as gone"


def test_R6_a_consumer_that_died_before_acking_gets_its_entry_back(client, stream):
    """The task's falsifier from the source's side: a bridge killed between the durable emit and
    the acknowledgement must find the arrival again, once, with its payload."""
    client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    sent = client.xadd(stream, {"payload": "a", "source": "probe"})

    delivered = read_new(client, stream, "bridge-1")
    assert [i for i, _ in delivered] == [sent]
    # The bridge dies here: the event was emitted durably, the ack never ran.

    # The replacement reads its own pending list under the SAME consumer name, which is what a
    # bridge with a fork-stable subscription name would do.
    again = read_pending(client, stream, "bridge-1")
    assert again == [(sent, {"payload": "a", "source": "probe"})]
    assert [i for i, _ in again] == [sent], "redelivery under a second id would defeat dedup"
