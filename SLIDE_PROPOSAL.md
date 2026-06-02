# Slides Proposal: Day 03 (Regression) and Day 04 (Modeling Project)

## Current State
- **Day 01**: Comprehensive slides covering EDA with pandas on SDSS dataset ✓
- **Day 02**: Well-structured slides on KNN classification ✓
- **Day 03**: Minimal slides (7 slides) - needs enhancement
- **Day 04**: Minimal slides (9 slides) - needs enhancement

---

## Day 03: Regression with scikit-learn

### Current Slides (7 total)
1. Title slide
2. Classification vs Regression comparison table
3. Linear Regression formula & code
4. MSE and R² metrics
5. Anscombe's Quartet
6. Diagnostic plots
7. Random Forest regression
8. Activity outline

### Proposed Enhancements

#### Missing Conceptual Slides (add 8-10 slides)
1. **"What is Regression?"** — Real-world examples (not just definitions)
   - Predicting brightness (z-magnitude) from colors
   - Continuous output vs categories
   - When to use regression
   
2. **"Why Linear Regression?"** — Intuition before equations
   - Fitting a line to data
   - Error/residuals concept
   - Minimize squared error (visual)
   
3. **"The Regression Pipeline"** — Parallel to classification pipeline (same pattern)
   - Scale features? (Yes, same reasons as KNN)
   - Train/test split (same)
   - Fit → Predict → Evaluate (same structure)
   
4. **"Residuals Deep Dive"** — What they mean
   - Residuals = prediction - actual
   - Plot residuals vs predicted
   - Look for patterns (fan shape = heteroscedasticity)
   - Look for structure (might be missing features)
   
5. **"Residuals Diagnostic Checklist"** — How to interpret
   - ✓ Centered on zero
   - ✓ Random scatter (no pattern)
   - ✓ Consistent spread (homoscedasticity)
   - ✗ Curved pattern → non-linear
   - ✗ Fan shape → variance increases
   
6. **"Comparing Models: Linear vs Random Forest"** — Same data, different approaches
   - Linear: interpretable, fast, assumes linear relationship
   - RF: flexible, captures non-linearity, harder to interpret
   - SDSS example: which performs better on z-magnitude prediction?
   
7. **"Why Does Redshift Help Regression?"** — Parallel to classification
   - Strong linear relationship with z-magnitude
   - But also hides structure (is the model just memorizing redshift?)
   
8. **"Per-Class Regression"** — Different models for each class
   - STAR colors behave differently than GALAXY
   - One model vs three models
   - Trade-off: flexibility vs data per model

#### New D2 Diagrams (3 diagrams)
1. **regression-pipeline.d2** — Parallel structure to ml-pipeline.d2
2. **residuals-diagnostic.d2** — Visual guide to interpreting residuals
3. **model-comparison.d2** — Linear vs Random Forest conceptual comparison

---

## Day 04: Modeling Project

### Current Slides (9 total)
1. Title slide
2. Full ML Loop diagram
3. New dataset introduction (molecular energy, 1275 features)
4. Start with a terrible model
5. Feature selection (RFE)
6. Cross-validation
7. PCA explanation
8. Activity outline (incomplete)

### Proposed Enhancements

#### Missing Conceptual Slides (add 10-12 slides)

1. **"Why This Dataset is Different"** — Transition from SDSS
   - SDSS: 17 features, clean, well-understood
   - Molecular energy: 1275 features, mostly noise
   - Real problem: feature explosion
   
2. **"High-Dimensional Data"** — What's the challenge?
   - More features = more parameters = easier to overfit
   - Curse of dimensionality
   - Visual: 2D scatter vs 1275D space (concept)
   
3. **"Overfitting vs Underfitting"** — The bias-variance tradeoff
   - Underfitting: too simple, misses patterns
   - Overfitting: too complex, memorizes noise
   - Goldilocks zone: generalize well
   
4. **"Why Start Terrible?"** — The diagnostic baseline
   - Build simplest model (Linear Regression on all features)
   - Expect it to fail
   - This tells you: "What needs improvement?"
   
5. **"Feature Selection is Art"** — Not just which features matter
   - RFE: iteratively remove least important
   - Domain knowledge: use physics intuition
   - Trade-off: more features vs model complexity
   
6. **"What is Cross-Validation?"** — Beyond train/test split
   - One split = one estimate, one number, prone to luck
   - K-fold: get k estimates, average them
   - More robust confidence in model performance
   
7. **"Dimensionality Reduction with PCA"** — Shrink feature space wisely
   - 1275 features → 50 PCs that explain 95% variance
   - PCA finds directions of maximum variance
   - Trade-off: lose interpretability, gain efficiency
   
8. **"The Iteration Loop"** — What to try next
   - Build baseline → Evaluate → Identify bottleneck → Try fix → Re-evaluate
   - Possible fixes: feature selection, regularization, different model, PCA, hyperparameter tuning
   - This IS the job
   
9. **"Common Pitfalls"** — What students will encounter
   - Using all features (overfitting)
   - Not scaling features
   - Tuning hyperparameters on test data (data leakage)
   - Claiming success on training accuracy alone
   
10. **"Your Activity Goals"** — Scaffolding
    - Goal 1: Establish baseline (bad linear model)
    - Goal 2: Clean features (remove noise)
    - Goal 3: Add cross-validation (confidence)
    - Goal 4: Iterate (pick one improvement)

#### New D2 Diagrams (4 diagrams)
1. **overfitting-underfitting.d2** — Model complexity vs performance curve
2. **feature-selection-flow.d2** — Steps in feature selection
3. **cross-validation-strategy.d2** — K-fold explained visually
4. **model-iteration-loop.d2** — How to improve: evaluate → diagnose → fix

---

## Design Consistency

### Color Scheme (from Nordic theme)
- Grays: `#767676`, `#1C1C1C`
- Orange: `#C46E3A` (explore/iterate)
- Blue: `#3D6B8E` (build/model)
- Green: `#5E9E7E` (evaluate)
- Red: `#8E3D3D` (error/problems)

### Style Guidelines
- Use same rounded rectangles as existing d2 diagrams
- Consistent stroke widths and colors
- Flow arrows show progression
- Callout boxes for key concepts

---

## Implementation Plan

1. **Week 1**: Enhance Day 03 slides (add 8-10 new slides + 3 d2 diagrams)
2. **Week 1**: Enhance Day 04 slides (add 10-12 new slides + 4 d2 diagrams)
3. **Testing**: Generate PDFs and verify rendering
4. **Refinement**: Adjust based on classroom feedback

---

## Files to Create/Modify

### Day 03 Enhancements
- `slides/day-03-regression.md` — Add new slides
- `slides/figures/d2/regression-pipeline.d2` — NEW
- `slides/figures/d2/residuals-diagnostic.d2` — NEW
- `slides/figures/d2/model-comparison.d2` — NEW

### Day 04 Enhancements
- `slides/day-04-modeling.md` — Add new slides + rewrite Activity section
- `slides/figures/d2/overfitting-underfitting.d2` — NEW
- `slides/figures/d2/feature-selection-flow.d2` — NEW
- `slides/figures/d2/cross-validation-strategy.d2` — NEW
- `slides/figures/d2/model-iteration-loop.d2` — NEW

---

## Success Criteria

✓ Each slide teaches one concept clearly
✓ Consistent narrative flow (why → what → how → when)
✓ D2 diagrams match existing visual style
✓ Align with corresponding activity notebooks
✓ Scaffolding appropriate for the task level
✓ Connection to SDSS dataset (Days 1-3) and molecular dataset (Day 4)
