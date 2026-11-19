# Correction of the test-message generator

The scripts that embedded random text messages built each message as

    "".join(random.Random(seed).choice(ALPHABET) for _ in range(n))

which creates a new random generator, with the same seed, for every character. Every test message was therefore one
character repeated. Exact recovery does not depend on the message, but distortion does, because the character fixes
the share of one bits in the payload.

All affected scripts now use one generator per message:

    (lambda _rng: "".join(_rng.choice(ALPHABET) for _ in range(n)))(random.Random(seed))

and every experiment that embeds a text message was run again with a fresh random message per image. The result files
in this repository are those of the reruns, and every number in the revised manuscript and its supplement comes from
them. Not affected: the matched steganalysis (random bytes), the random-bit check of Supplementary Note 2, the
encryption test and the synthetic checks.

Rerun: the 1,000-pair corpus (Table 2, Fig. 4a,b, Supplementary Tables S1, S4, S13), the bin-centre comparison on the
same pairs and messages (`evaluation/10_bincentre_all.py`, new), robust mode (S6), unseen prompts (S7), the numerical
checks (Note 3, S8 to S11, Fig. S2), processed covers and photographs (Table 3, S12), six domains (S16), texture
photographs (S17), the ablation (S18), the gain split and offsets (S19, S20), capacity at equal quality (S22), and the
"DEDS, corpus" rows of the detectability table (`steganalysis/code/15_rescore_rerun.py`, new; the trained detectors are
unchanged). The conclusions are unchanged; with random messages the fine-tuned covers give the better image in 999 of
1,000 corpus pairs and carry more at equal quality than their baseline pairs in at least 198 of 200 pairs.

`generation/12_timing_single_process.py` and `steganalysis/code/01_embed_training_stego.py` carry the same fix; the
first only times the embedder and the second built training images that the paper does not use.
