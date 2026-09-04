# Business Hours Risk Policy

The system evaluates transactions differently depending on whether they occur during business
hours (weekdays 09:00-18:00) or outside them. Data analysis showed that off-hours — especially the
04:00-09:00 window — carry very low transaction volume, but a higher fraud rate than the rest of
the day.

Two different correction approaches are applied for this reason: a volume/confidence-based
approach dampens the score for off-hours transactions, on the assumption that statistics built on
sparse data are less reliable; a fraud-rate-calibrated approach instead boosts the score for
off-hours transactions, based on the actually measured fraud rate. Both methods are computed and
compared separately — the system does not assume a single "correct" direction, it transparently
presents the results of both principles.

No correction is applied to transactions within business hours; the multiplier is always 1.0.
