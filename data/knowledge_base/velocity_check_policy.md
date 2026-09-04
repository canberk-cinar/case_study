# Velocity (Rapid Repeat Transaction) Policy

The same card transacting again within a very short time is a classic signature of card testing
or automated fraud attacks. The system tracks the time elapsed since a card's previous
transaction.

Repeat transactions within less than 60 seconds are flagged at HIGH severity and routed to review
(REVIEW). A stronger pattern is the combination of "address change + rapid repeat": a new billing
address never used before on this card, used again within 300 seconds — resembling the
address-hopping pattern fraud rings use to test stolen card numbers against multiple drop
addresses — and this triggers the block (BLOCK) action at CRITICAL severity.

If the card has no prior transaction at all (a first transaction), the velocity check is
meaningless and is not applied.
