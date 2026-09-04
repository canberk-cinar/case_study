# Device Fingerprint Policy

Transactions where the device information (DeviceInfo) is missing, or carries only a generic
operating-system label (e.g. "Windows", "iOS Device", "MacOS") rather than a specific device
model, are treated as transactions whose device fingerprint could not be verified.

On its own this carries low risk, but combined with a high-value transaction it is flagged at
MEDIUM severity and routed to review (FLAG). Additionally, a new device and a new billing address
being seen for the same card AT THE SAME TIME — a classic account-takeover signature — is flagged
at HIGH severity.

This policy is based only on the device information being generic/missing; it does not target any
specific device brand or model.
