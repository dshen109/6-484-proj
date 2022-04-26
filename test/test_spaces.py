from deep_hvac import spaces

from numpy.testing import assert_array_equal

from unittest import TestCase


class TestMultiDiscrete(TestCase):

    def test_sample(self):
        space = spaces.MultiDiscrete(
            nvec=[1, 2, 10], starts=[1, 10, 200], seed=1)
        sample1 = space.sample()
        sample2 = space.sample()
        assert_array_equal([1, 11, 201], sample1)
        assert_array_equal([1, 10, 204], sample2)

    def test_contains(self):
        space = spaces.MultiDiscrete(nvec=[1, 2, 10], starts=[1, 10, 200])
        self.assertTrue(space.contains([1, 11, 205]))
        self.assertTrue(not space.contains([-1, 11, 205]))
        self.assertTrue(not space.contains([1, 11, 300]))
