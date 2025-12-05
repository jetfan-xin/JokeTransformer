
# Procedure
On top 5000 combinations, for each combo, generate 100 high-quality jokes.
# Try different LLMs
| **Group** | **Model**             | **Speed (/100 jokes/GPU)** | **Jokes / Prompt**   | **Limits & Pricing**                      | **Collected** | **LLMs as Judge** |
| --------- | --------------------- | -------------------------- | -------------------- | ----------------------------------------- | ------------- | ----------------- |
| A         | Llama-3.1-70B         | 270 s                      | 20                   | -                                         | 32,531        | 2.5/10            |
| B         | Llama-3.1-8B          | ~60 s                    | 20                   | -                                         | 3,879         | 3/10              |
| C         | **Qwen3-30B-A3B**     | 15 min                     | 20 | -                                         | 7,231         | **7.5/10**        |
| D         | **DeepSeek-Chat**     | -<br>(no rate limit)       | 10                   | €1 ≈ 24k jokes                            | 23,567        | **7.5/10**        |
| E         | Gemini-2.5-Flash      | - <br>(rate limited)       | 20 (75%)<br>50 (25%) | Free Tier: 250 reqs/day<br>(~12.5k jokes) | 6,177         | 4.5/10            |
| F         | Gemini-2.5-Flash-Lite | - <br>(rate limited)       | 50                   | Free Tier: 1,000 reqs/day<br>(~50k jokes) | 2,128         | 3.5/10            |
| G         | Gemini-2.5-Pro        | - <br>(rate limited)       | 50                   | Free Tier: 50 reqs/day<br>(~2.5k jokes)   | 2,487         | 7/10              |
| **Total** | -                     | -                          | -                    | -                                         | **78,000**    |                   |
## LLMs as judge
Used used GPT and Gemini as anonymous judges to evaluate jokes from different models.

For each model, evaluated **50 random samples**  
(from `/uhh-ias-ml/data/llm_jokes/stats/model_joke_samples.json`).

### Prompt
```
TASK:
Evaluate the comedic quality of seven groups of jokes labeled A through G.
Each group contains many joke samples. You must evaluate the entire group, not each joke individually.

⸻

SCORING RUBRIC (1–10):

Use the following criteria when scoring each group:
	1.	Joke Craftsmanship
	•	Clear setup and punchline
	•	Word economy; avoids rambling
	•	Misdirection, twist, or clever turn
	•	Internal logic that makes sense
	2.	Originality & Creativity
	•	Avoids generic or recycled joke templates
	•	Punchlines are fresh, surprising, or inventive
	3.	Consistency Within the Group
	•	Fewer “dead” or unfinished jokes
	•	Jokes feel like they could come from the same competent writer
	•	Repetition is avoided or intentionally stylistic
	4.	Voice & Tone
	•	A distinct comedic persona or style elevates the group
	•	Tone holds steady (absurd, dark, clean, witty, surreal, etc.)
	5.	Use of Edgy / Sensitive Material
	•	Edgy jokes must still be clever, not just shocking
	•	If they rely on slurs, trauma, tragedy, etc., they must justify the topic with real craft—not cheap provocation

Interpretation of Score:
	•	9–10: Near professional-level comedy writing
	•	7–8: Strong set with a consistent voice, mostly effective jokes
	•	5–6: Mixed quality; noticeable misses or weak formulas
	•	3–4: Mostly weak; generic, incoherent, or poorly constructed
	•	1–2: Fundamentally broken or non-jokes

⸻

YOUR OUTPUT FORMAT:

For each group A–G:
	1.	Score (1–10)
	2.	Short justification (4–7 sentences):
	•	Mention strengths
	•	Mention weaknesses
	•	Evaluate consistency, joke construction, originality
	•	Call out recurring problems (lazy puns, non-sequiturs, edgy-without-payoff)
	3.	One-sentence summary describing the “overall vibe” of the group

Example output structure:

Group A: 4/10  
Explanation: …  
Summary: …


⸻

DATA TO EVALUATE:
{jokes here}
```

### EvaluationResults
#### Group A: 2.5/10
- GPT: **3/10**
	**Explanation**: Most jokes are simple Q&A puns, but many land as stock or generic (“banana wasn’t peeling well,” “can’t opener”). There’s noticeable repetition of weak templates (“paws-itive,” therapy/beauty parlor setups) which makes the set feel padded rather than stylistic. Several edgy premises (abortion, Jew/oven, leper, kidnapping) are handled without much wit, so they read as attempts at shock more than crafted dark humor. Internal logic is often loose or lazy, with punchlines that don’t specifically pay off the setup (“coffee, penis, woman” joke never uses the combo). There _is_ a basic sense of how a joke structure works, but the execution is very first-draft.
	**Summary**: A pile of dad-joke puns with occasional edginess, but not enough craft to justify how often it tries to be dark or clever.
- Gemini: **2/10** 
	**Explanation**: This group relies almost exclusively on the lowest form of word-association puns (“paws-itively,” “filling good time,” “word-smith”). The craftsmanship is lazy, often forcing a setup just to accommodate a weak play on words that has been heard countless times before. While most are harmless “Dad jokes,” the sudden inclusion of a Holocaust joke (Jew/oven) is a catastrophic tonal failure; it lacks any cleverness to justify the edge, coming across as mere shock value amidst a sea of G-rated puns. The logic in jokes like the “abortion doctor” is also nonsensical rather than funny. 
	**Summary:** A collection of groan-worthy, bottom-tier puns with a jarring and poorly executed attempt at offensive humor.
#### Group B: 3/10
- GPT: **4/10**
	**Explanation**: This group has a clearer feel for basic joke forms and some decent wordplay (“depresso,” “grizzly illusionist,” “award for worst gardener from a scarecrow”). However, it’s weighed down by a lot of muddled punchlines and anti-jokes that don’t resolve cleanly (“period”/coffee mugged, random anti-gravity callbacks that don’t fit the setup). The voice wanders between standard dad-joke punning and surreal non sequiturs without committing to either, so the set feels unfocused. Repeated motifs (post, mood ring, hokey pokey, ocean) don’t build into a bit; they just feel like recycling. Overall, it’s slightly stronger than A in craft but still full of misses.
	**Summary**: Feels like a joke notebook full of half-baked puns and stray ideas, with only a few that actually click.
- Gemini: **2/10** 
	**Explanation:** Similar to Group A, this set suffers from a severe lack of originality, relying on ancient templates like the “Hokey Pokey” or “Depresso” coffee jokes. The personification jokes (stairs, soap) are childish and lack a surprising punchline, often simply stating a property of the object (soap is clean, stairs go up). There is no distinct comedic voice here; it reads like a randomized output of a 1990s joke book. While it avoids the offensive missteps of Group A, it is aggressively bland and uninspired. 
	**Summary:** An unimaginative assortment of recycled wordplay and flat one-liners that lacks any comedic spark.
#### Group C: 7.5/10
- GPT: **7/10**
	**Explanation**: This set has a distinct voice: self-aware, slightly surreal, and comfortable anthropomorphizing everything from beds to hair to hands. Many jokes lean into narrative or character instead of simple Q&A structure, which creates a more original feel (“my hand started a podcast,” “my bed caught me lying”). There are recurring motifs (hippo/Zippo, punchline, letters) that are reimagined in different ways instead of copy-pasted, showing some range. A few edgy topics (Alzheimer’s, penis, funerals) are treated with more conceptual humor than pure shock, though some will read as glib rather than insightful. Not every line lands, but the overall craftsmanship and consistency are notably higher than in A and B.
	**Summary**: A quirky, introspective, slightly dark comedian brain monologuing—uneven, but with a real style and some genuinely clever turns.
- Gemini: **8/10**
	**Explanation:** This group demonstrates a strong, consistent voice characterized by absurdist personification and meta-humor. Jokes like the jeans writing poetry about digestive trauma or the bed sensing guilt show high creativity and a fresh perspective on mundane objects. The self-referential jokes about “punchlines” and “definitions” display wit and intelligence. While a few entries are slightly wordy, the hit rate is high, and the writer successfully elevates simple premises into surreal, memorable bits. 
	**Summary:** A smart, surreal, and creative set that successfully blends meta-comedy with vivid, absurdist imagery.
#### Group D: 7.5/10
- GPT: **8/10**
	**Explanation**: This group shows strong control of classic joke mechanics—clear setups, logical misdirection, and precise wording (“outstanding in his field but lacked drive,” “B-negative like my attitude about group projects”). The voice is witty, mildly absurd, and consistently clever without leaning too hard on shock value. Even when touching on edgier subjects (ISIS acronym, North Korea roommate, unemployment), the humor comes from wordplay or metaphor rather than cruelty. There are a few softer or more whimsical pieces that feel like lighter sketches, but they’re still coherent jokes, not filler. The overall hit rate is high, and the jokes feel like they could plausibly belong in a seasoned writer’s set.
	**Summary**: Smart, polished wordplay with a lightly absurd edge—like a professional joke writer having fun with a long prompt list.
- Gemini: **7/10**
	**Explanation:** This group displays solid craftsmanship, particularly in how it twists common idioms into fresh punchlines (e.g., the scarecrow being “outstanding in his field” but lacking drive, or the unicycle taking a stand). The character work is effective, particularly the Batman and “Nerf or Nothing” jokes, which rely on cultural observation rather than just wordplay. The humor is consistent and witty, though it occasionally leans on slightly dated references (ISIS). Overall, it feels like the work of a competent comedy writer who understands structure and misdirection.
	**Summary:** A witty and well-constructed collection that cleverly plays with language and character tropes.
#### Group E: 4.5/10
- GPT: **5/10** 
	**Explanation**: This set is wildly mixed: some jokes are classic, well-structured one-liners (“OJ factory couldn’t concentrate,” “running late doesn’t count as exercise”) while others are awkward mini-scenarios or sentences that barely qualify as jokes. There’s a lot of topic whiplash—Trump, coronavirus, suicide, tampons, diapers/politicians—without a strong unifying comedic persona to ground it. Edgy or topical material often feels like it’s there because the subject is “hot,” not because the punchline is particularly sharp. Several setups ignore their own keyword combos or bury them in clunky exposition, which hurts economy. Still, enough jokes are structurally sound and occasionally clever to keep it in middle territory.
	**Summary**: A grab bag of one-liners where solid dad jokes and timely quips share space with clunky, try-hard edginess.
- Gemini: **4/10**
	**Explanation:** This group attempts to blend standard puns with edgier topics (drugs, sex, politics) but largely fails to stick the landing. The “depresso” joke appears again, indicating a lack of originality, and the “tampon/period” jokes are repetitive and rely on the mere mention of the subject rather than a clever twist. The “pussy/cat” double entendres are juvenile and lack the sophistication seen in Groups C or D. It feels like a disjointed mix of children's joke structures applied to adult themes without the necessary nuance. 
	**Summary:** A mediocre attempt at adult humor that relies too heavily on lazy innuendo and repetitive premises.
#### Group F: 3.5/10
- GPT: **4/10**
	**Explanation**: There’s some familiarity with standard joke formats (“put my foot down,” flamingo bit, fraction/numerator line), but the set leans heavily on stale wife/“right woman” jokes and Irishman stereotypes. Sexual and edgy material (mafia + “pussy,” vibrator/banana, poo sculpture) is often used more for provocation than for a genuinely smart angle. Many pieces are just premises or images with a weak or missing twist, so they read more like stray lines than crafted jokes. Repetition around specific combos (right woman, banana/vibrator, mafia/pussy) doesn’t build into a coherent bit, it just feels padded. The voice is inconsistent, swinging from dad-jokey to raunchy without a clear persona.
	**Summary**: A lumpy mix of dated spouse jokes, raunchy gags, and a few decent wordplays that can’t quite pull the set above “pretty weak.”
- Gemini: **3/10**
	**Explanation:** The voice here is heavily reliant on dated, "Boomer-era" tropes: the nagging wife, the bad cook, and lazy ethnic stereotypes (Irish/potatoes). The "mafia/pussy" joke is an example of "edgy" material failing—it is long-winded and the payoff is a cheap, predictable pun. The jokes lack internal logic (the fraction joke is a mess) and the "I hate my wife" energy makes the tone feel bitter rather than funny. It scores slightly higher than A or B only because a few surreal bits (banana/vibrator) show a flicker of an attempt at modern absurdity.
	**Summary:** A tired collection of "ball-and-chain" complaints and forced innuendos that feels decades out of date.
#### Group G 7/10
- GPT: **8/10**
	**Explanation**: This group has a clear, confident comedic voice, especially around recurring mini-bits like “my daughter Anna, the drummer” and the friend striking out with girls. Many jokes show good structure and timing—setups that lead logically to a twist (“unicycle = late for clown court,” “friend describes himself in three words: not good at math”). The running callbacks (Anna the drummer, nuts, weasels/bars, skydiving) feel intentional and character-driven rather than lazy repetition. Edgy material is light and mostly playful; there’s some innuendo and mild cynicism about marriage/sex, but it’s framed as relatable stand-up rather than shock humor. Overall, it reads like material from a competent club comic with a taste for silly premises and character bits.
	**Summary**: Feels like a tight, modern stand-up set built around recurring characters and smart twists on familiar setups.
- Gemini: **6/10**
	**Explanation:** This group shows promise through the use of specific recurring characters ("Anna the drummer") and elaborate scenarios, which adds a layer of world-building missing from the other groups. The "German lightbulb" and "Weasel/Pop" jokes show an ability to construct narrative jokes rather than just simple Q&A puns. However, the quality is inconsistent; some jokes land well with a surreal twist (horse/mayo), while others are a bit clunky or rambling. It feels like a writer testing out material—some of it is unique and funny, but it needs tightening. 
	**Summary:** An above-average set with some genuinely creative narrative elements and specific character humor, weighed down by a few duds.