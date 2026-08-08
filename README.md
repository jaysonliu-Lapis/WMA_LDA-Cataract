# PROP for cataract comorbidity classification

This repository packages the exact real-data cohort, ResNet-50 features, and R implementation used for the PROP cataract experiment. PROP is a model-averaging discriminant classifier: it screens variables, builds random-subspace DSDA base learners, learns simplex-constrained ensemble weights by quadratic programming, and classifies using the weighted discriminant score.
