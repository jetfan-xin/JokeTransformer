# Plot Data Overview

## 1. POS-based Tag Type Distribution

#### Files
- tag_type_count.csv
- tag_type_count.html / .png

#### Description
Classifies all tags into POS categories:
- NOUN
- PROPN
- VERB
- ADJ
- MISC

The bar plot displays both absolute counts and the percentage share.

## 2. oke Length Statistics

#### Files
- joke_length_chars.csv — character lengths of all jokes
- joke_length_chars.html / .png — histogram (bin width = 10)

#### Description
Computes the character length of each joke and plots the distribution.

The plot also includes:
- cumulative percentage
- heatmap-style coloring by cumulative percent
- highlighted 90% or 95% length threshold

## 3. Number of Tags per Joke
#### Files
- tag_count_distribution.csv
- tag_count_distribution.html / .png

#### Description
Counts how many tags each joke has (1, 2, or 3).

The bar chart shows:
- total count
- percentage for each category
- hover tooltip with full details

## 4. Tag Frequency (Top-N)
#### Files
- tag_frequencies_full.csv
- tag_frequencies_top30.(csv/html/png)
- tag_frequencies_top100.(csv/html/png)
- tag_frequencies_top500.(csv/html/png)
- tag_frequencies_top1000.(csv/html/png)

#### Description
Ranks all tags by frequency.

Creates bar charts for the top-N tags (N = 30, 100, 500, 1000), with:

- log-scaled y-axis
- labels for frequency

## 5. Full Tag Frequency Distribution
#### Files
- tag_frequencies_full.csv
- tag_frequency_distribution_log.html / .png

#### Description
Uses logarithmic binning (powers of two) to show how many tags fall within each frequency range.

Each bar includes:
- 2^n bin range (e.g., 4–7, 32–63)
- count
- percentage
- cumulative percentage
- color based on cumulative (heatmap style)

This reveals the long-tail distribution clearly.