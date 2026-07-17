# Data Analytics -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Any SQL or analysis logic is correct -- right join type, correct aggregation grain, no double-counting -- and any chart or visualization recommendation avoids deceptive framing (no truncated axes, no cherry-picked ranges). Statistical claims are appropriately hedged (correlation is not stated as causation), and every finding is traceable to the specific numbers given in the prompt rather than invented.
Example character: a query with the correct join and grain explained, a chart recommendation with the axis explicitly kept honest, and a finding that cites the specific input numbers.

## 7 -- Strong
The core analysis is correct and the findings are usable, but there is one minor query or logic gap -- an edge case in a join, for instance -- or the chart-framing caveat is left unstated. Nothing here would silently corrupt the result.
Example character: a correct query with a sound finding that does not explicitly address how the chart avoids visual distortion.

## 5 -- Adequate
The analysis is directionally correct but the query has a plausible bug (wrong aggregation grain causing double-counting) or the chart recommendation is not checked for distortion before being suggested.
Example character: a query that looks reasonable but would double-count rows on a one-to-many join.

## 3 -- Weak
The SQL or logic has a clear correctness bug (wrong join type, missing GROUP BY column), correlation is stated as causation, or a trend is asserted that the given data does not support.
Example character: a query missing a needed GROUP BY that would silently produce wrong aggregates.

## 0 -- Unusable
Presents fabricated numbers as if they were query output, or recommends a chart design that is actively deceptive (truncated axis to exaggerate a trend) as best practice. Empty or off-task.

## Unit-specific red flags (deduct hard)
- SQL or logic bug that would silently double-count or drop rows (wrong join type, missing GROUP BY column).
- Fabricated numbers presented as if they came from the given dataset.
- Chart design that would visually mislead (truncated y-axis, cherry-picked time window) recommended without caveat.
- Correlation stated as causation.

## Unit-specific excellence markers
- Query or logic correct for the actual grain and join semantics needed.
- Chart or visualization choice avoids known distortion patterns.
- Statistical uncertainty or the correlation-versus-causation distinction called out explicitly.
- Findings traceable to the specific numbers given in the prompt.
