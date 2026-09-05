# New Card, High-Value First Transaction Policy

A card's very first observed transaction (zero prior transaction history) being high-value means a
large risk is being taken with no historical data to verify the card's behavior. This is flagged
at CRITICAL severity and triggers the block (BLOCK) action.

Additionally, high-value transactions occurring during the weekend and the low-volume hour window
(04:00-09:00) receive a separate MEDIUM-level flag (FLAG), based on the observation that
combining two individually weak contextual signals (weekend + low volume) is a stronger risk
indicator than either alone.

The high-amount threshold is derived from the data's own percentiles (roughly the top 5%); it was
not chosen by looking at the actual fraud label.
