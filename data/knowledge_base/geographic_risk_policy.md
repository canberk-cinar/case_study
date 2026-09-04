# Geographic Risk Policy

The billing region/country code (addr2) is the strongest risk signal measured in the system.
99.2% of the data belongs to a single domestic region code; transactions with a different
(foreign) code have a fraud rate about 4.26x higher than domestic transactions. Transactions with
a missing region code have an even higher fraud rate than foreign ones (about 4.91x) — which is
why the policy treats "foreign" and "missing" as two SEPARATE risk tiers rather than merging them
into one "foreign" category.

Two methods are applied: a fixed policy multiplier (only for transactions with a definitively
known foreign code, 2.0x — missing data is left untouched, because a label-free policy cannot
assume "unknown" means "risky") and a fraud-rate-calibrated multiplier (domestic=1.0,
foreign≈4.26, missing≈4.91).

Other geographic candidates (physical distance, email domain suffix) were tested but not used
because they were not found to be reliable.
