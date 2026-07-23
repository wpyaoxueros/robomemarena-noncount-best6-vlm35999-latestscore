# Task4 v4 Anchor Frame-0 Attempt Is Invalid

The v4 job was cancelled before scoring. It enabled the three historical
release-anchor rules but omitted their historical frame indices, so the
runtime defaulted to frames `0/0/0`.

The old seed108 trace records frames `20/20/50` for close-top to open-middle,
close-middle to open-bottom and close-bottom to open-top-again respectively.
This attempt is excluded from results and is retained only as an auditable
configuration error. The corrected replay is v5.
