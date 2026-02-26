import unittest

from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.encoding import encode_features, get_encoder


class EncodingTests(unittest.TestCase):
    def test_encode_shape_and_cache(self) -> None:
        domain = build_comp_car_domain()
        rows = domain.sample_lhs(3, __import__("numpy").random.default_rng(1))
        a, cols = encode_features(domain, rows)
        self.assertEqual(a.shape[0], 3)
        self.assertEqual(a.shape[1], len(cols))
        self.assertIs(get_encoder(domain), get_encoder(domain))


if __name__ == "__main__":
    unittest.main()
