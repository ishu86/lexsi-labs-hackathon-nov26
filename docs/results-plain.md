# Results in plain terms (final; 2026-11-16 06:50 UTC)

Wording the owner asked to keep. Exact rates, intervals and p-values are in `results/*_tables.json`.

## What we did, in one line

We took a safe model (Cohere's Tiny Aya Global), taught it a harmless new job, saw that the teaching made it less safe, tested which repair tools make it safe again without forgetting the job, and then checked the chosen repairs on questions nobody had used for choosing.

## How to read the scores

Every score is "out of 100".

- **Harmful:** out of 100 harmful questions, how many answers contain harmful content. Lower is better. Graded by a judge that agrees with human raters on 94 of 100 answers (public judges on the same answers: GPT-4 92, Llama-3 90).
- **Refuses:** how many it clearly says no to. Higher is better.
- **Rest:** answers that neither refuse nor contain harmful content.
- **Skill:** out of 100 fresh questions on the new job (naming the intent of a message in six languages), how many it gets right.
- **Knowledge:** out of 100 general-knowledge questions in five languages.
- **Noise:** on 160 questions each number can move by about 3 by chance; on the 600 multilingual questions by about 1.5.
- Answers are up to 256 tokens long. About 7 in 100 answers do not finish by then; they do not finish by 512 either, and the grades are the same at both lengths.

## Result 1: harmless training broke the safety

Training data: 6,400 intent examples and 5,600 ordinary question-and-answer pairs (Dolly). No harmful content.

| model | harmful, English (HarmBench) | harmful, four other languages | skill |
|---|---|---|---|
| original | 14 | 9 | 0 |
| after training (`ft_v3`) | 42 | 24 | 71 |

Far outside chance on both sets. The three other fine-tunes we made show the same (23 to 35 harmful on HarmBench). In small trial runs, English Q&A data eroded safety and multilingual Q&A data barely did, and more training kept eroding safety after the skill had stopped improving.

## Result 2: the repairs, on questions nobody chose on

| repair on `ft_v3` | what it does | harmful, English | harmful, other languages | refuses, English | skill | knowledge |
|---|---|---|---|---|---|---|
| none | | 42 | 24 | 36 | 71 | 51 |
| `wise_ft` 0.5 | moves the weights halfway back to the original | 17 | 12 | 77 | 70 | 50 |
| `wise_ft` 0.53 | a little more than halfway | 16 | 12 | 79 | 66 | 50 |
| `safe_delta` 0.8 | keeps only part of the training | 23 | not run | 63 | 74 | 51 |
| `steer_bias` L18 | a push toward "no", written into the weights | 27 | 10 | 57 | 71 | 41 |
| steering L16, strength 9 (reference, not a weight change) | the same kind of push, added while the model runs | 10 | 7 | 81 | 72 | 45 |
| original, for comparison | | 14 | 9 | 84 | 0 | 51 |

- **Halfway weight interpolation works.** It takes harmful answers from 42 to 17 in English (the original is at 14; that gap is inside noise) and from 24 to 12 in the other languages, and keeps the skill and the knowledge.
- **Steering is still ahead on harm** (10 and 7), and keeps the skill, but it loses 6 points of multilingual knowledge and is not a change to the weights.
- **The practice-round favourite paid for it.** `wise_ft` 0.53 scored best on the practice questions and then lost 5 points of skill on fresh items.
- **Tools that "repair" by erasing the job** (`safe_lora`, `svr`, strong `wise_ft`) were dropped in the practice rounds: they brought skill down to 0-33.

By language (harmful out of 100: Arabic, Bengali, Swahili, Hindi): original 8, 13, 6, 9; unrepaired 23, 24, 18, 32; `wise_ft` 0.5: 12, 14, 8, 15; steering: 4, 4, 3, 16. Hindi comes from a different source and is the hardest for every repair.

## Result 3: do the picks carry over to other fine-tunes?

The same two settings, applied unchanged to three fine-tunes they were not chosen on (HarmBench, out of 100):

| fine-tune | unrepaired harmful / skill | `wise_ft` 0.5 harmful / skill | steering harmful / skill |
|---|---|---|---|
| `ft_v1_e1` (lightest training) | 33 / 67 | 15 / 58 | 7 / 67 |
| `ft_v2` | 23 / 72 | 11 / 72 | 7 / 71 |
| `ft_v3_s600` (early snapshot) | 35 / 70 | 17 / 66 | 10 / 69 |

The harm reduction carries over everywhere. The "no skill cost" of halfway interpolation does not: on the lightest fine-tune it costs 9 points of skill. That one dial has to be adjusted per model.

## Result 4: who picked best

Six pickers each got 6 tries on practice questions. Best practice score after 2, 4 and 6 tries:

| picker | after 2 | after 4 | after 6 | its pick |
|---|---|---|---|---|
| fixed rules from a weight diagnosis | 92 | 92 | 92 | `wise_ft` 0.5 |
| Aya agent, Aya agent with examples | 92 | 92 | 92 | `wise_ft` 0.5 |
| random, typical run | 82 | 88 | 90 | |
| random, the run we got | 80 | 94 | 94 | `wise_ft` 0.53 |
| steering ladder (reference) | 90 | 90 | 93 | steering L16 s9 |
| DeepSeek agent | 80 | 82 | 84 | `safe_delta` 0.8 |

A diagnosis plus a simple rule was enough. No LLM agent beat the rule or the random floor. Each picker ran once on one model, so we do not rank them.

## How we know the grader is sound

- First rubric ("did the model go along?") agreed with humans on only 62 of 100 public answers. We replaced the harm number with a rubric that asks "does the answer contain harmful content?": 94 of 100 on a half of the set that was not used to write it.
- The agents had first been given feedback from the weaker rubric; we re-scored and let them choose again. Their picks are reported under the corrected feedback.
- "Refuses": recognised every human-labelled refusal (123 of 123); called 12 of 141 full answers a refusal.
- Other languages (soft check, translated answers with human labels): agrees with humans on 88 of 100 (Arabic 96, Hindi 90, Swahili 88, Bengali 83).

## What we are not sure about

- One model family (Tiny Aya Global, 3.35B), one training run per fine-tune, one run per picker.
- The questions are asked directly, plus low-resource languages; no crafted jailbreak attacks were tested.
- The judge was not checked by humans on the small model's own answers in other languages.
- Azure's content filter blocked 4-6 in 100 judge calls; those answers are left out and lean harmful.
- Stacked repairs (interpolation plus steering) were not tried.
- `ft_v3` missed our own entry bar by one point of skill (74 against 75 on the practice items).
