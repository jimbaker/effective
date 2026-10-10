"""A claim's step handles are released with the claim's SDK ctx."""

import gc
import weakref

from absurd_sdk import StepHandle

from effective.engines.absurd import ConcurrentAbsurdCtx, SdkCtx
from effective.keys import Key


class _Sdk:
    """The two public SDK calls an adapter makes, over an in-memory store."""

    def begin_step(self, name: str) -> StepHandle:
        return StepHandle(name=name, checkpoint_name=name, done=False)

    def __init__(self) -> None:
        self.persisted: list[object] = []

    def complete_step(self, handle: StepHandle, value: object) -> object:
        self.persisted.append(value)
        return value


def test_a_claims_step_handles_are_released_with_its_sdk_ctx():
    sdk = _Sdk()
    SdkCtx(sdk).step(Key.parse("step;tool:a"), lambda: 1)
    ConcurrentAbsurdCtx(sdk).step(Key.parse("step;tool:b"), lambda: 2)
    released = weakref.ref(sdk)
    del sdk
    gc.collect()
    assert released() is None


def test_the_sdk_is_handed_its_own_copy_of_a_step_value():
    sdk, value = _Sdk(), {"v": [1]}
    assert SdkCtx(sdk).step(Key.parse("step;tool:a"), lambda: value) is value
    assert sdk.persisted == [value]
    assert sdk.persisted[0] is not value
