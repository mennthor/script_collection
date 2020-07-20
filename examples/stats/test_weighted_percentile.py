"""
Note: This example must be reviewed, it does not yield expected behaviour.
"""

import numpy as np
from script_collection.stats.tools import weighted_percentile

# Unweighted must be equal
test = np.random.uniform(0, 20, size=100)
print("Test unweighted case:")
print("numpy: ", np.percentile(test, 20, interpolation="lower"))
print("here:  ", weighted_percentile(test, 20))

# Weighted analytic test case
print("\nTest weighted case:")
test = np.arange(11) + 1.
weights = np.ones_like(test)
weights[5] = 2.
weights[6:] = 3.

print(test)
print(weights)

print("Unweighted is: ", weighted_percentile(test, 50.), ", should be 6.")
print("Weighted is: ", weighted_percentile(test, 50., weights=weights),
      " should be 8.")
