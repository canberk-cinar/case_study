# Trusted Entity Policy — An Unexpected Finding

The naive expectation was that a card with a long transaction history would be more trustworthy,
and its score could therefore be dampened. However, data analysis showed the EXACT OPPOSITE: while
new/thin-history cards have a low fraud rate (2.4-2.6%), cards with 101-1000 prior transactions
reach the highest fraud rate (3.91%).

A likely explanation is that the card identifier, at this scale, represents a shared "bucket"
value rather than a single real customer. Because of this, a naive correction based on the
"long history = trust" assumption ACTUALLY MADE the system's performance WORSE when tested — it
disrupted the ranking of the top risk bucket. A fraud-rate-calibrated correction in the opposite
direction produced the correct result instead.

This policy is a concrete example of why a business rule that "sounds reasonable" should not be
applied without testing it against the data.
