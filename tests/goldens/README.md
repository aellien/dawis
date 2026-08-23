# Goldens

Recorded outputs that pin DAWIS behaviour, plus `provenance.toml` recording
for each one: the `legacy-seeded` commit it came from, the seed, the
tolerances, and who regenerated it, when and why.

Goldens live in plain git, no LFS. Revisit that only if they grow past
about 100 MB.

They are a ratchet, not a shrine. Through the 1.0 refactor no golden is
regenerated, because nothing should move. When later science work does move
one, it is regenerated in a commit whose message says why.

Empty for now: Tier 1 goldens are recorded by #36, Tier 2 by #37.
