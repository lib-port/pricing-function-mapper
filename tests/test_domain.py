import unittest

from pricing_mapper.domain import build_comp_car_domain, canonicalize_comp_car_input


class DomainTests(unittest.TestCase):
    def test_canonicalize_bounds(self) -> None:
        row = {
            "driver_age": 10,
            "years_licensed": 50,
            "vehicle_year": 1800,
            "vehicle_value": 1,
            "annual_km": 999999,
            "claims_5y": -1,
            "convictions_5y": 9,
            "postcode_risk": -1.0,
            "theft_risk": 9.0,
            "excess": -5,
            "usage": "private",
            "parking": "garage",
            "hire_car": "none",
            "windscreen": "no",
            "rating": "market",
        }
        out = canonicalize_comp_car_input(row)
        self.assertEqual(out["driver_age"], 17.0)
        self.assertEqual(out["years_licensed"], 1)
        self.assertEqual(out["vehicle_year"], 1998)
        self.assertEqual(out["vehicle_value"], 2000.0)
        self.assertEqual(out["annual_km"], 60000)
        self.assertEqual(out["claims_5y"], 0)
        self.assertEqual(out["convictions_5y"], 6)
        self.assertEqual(out["postcode_risk"], 0.0)
        self.assertEqual(out["theft_risk"], 1.0)
        self.assertEqual(out["excess"], 0)

    def test_domain_overrides(self) -> None:
        domain = build_comp_car_domain(
            {
                "continuous": {"vehicle_value": {"low": 5000, "high": 100000}},
                "integers": {"excess": {"low": 100, "high": 3000}},
                "categorical": {"usage": ["private", "business"]},
            }
        )
        c = [v for v in domain.continuous if v.name == "vehicle_value"][0]
        i = [v for v in domain.integers if v.name == "excess"][0]
        k = [v for v in domain.categorical if v.name == "usage"][0]
        self.assertEqual((c.low, c.high), (5000.0, 100000.0))
        self.assertEqual((i.low, i.high), (100, 3000))
        self.assertEqual(k.levels, ["private", "business"])


if __name__ == "__main__":
    unittest.main()
