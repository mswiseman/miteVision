# miteVision 

The code on the jupyter notebook is not yet comprehensive since the real training notebook is 1Gb+ and I need to triple-check for API call removal. Regardless, you should get an idea for the process and I'd be happy to share the link on a request-basis until I have time to clean it up. 

In the meantime...

## Sex differentiation model (class balancing augmented with synthetic images)

[Link to model hosted on Roboflow](https://universe.roboflow.com/gent-lab/tssm-balanced-and-gendered)

Six classes:
* Adult_female
* Adult_male
* Dead_mite
* Juvenile
* Viable_egg
* Egg_cast

## Model best for acaricidal or ovicidal assays

[Link to model hosted on Roboflow](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/26)

Four classes:
* Mite
* Dead_mite
* Viable_egg
* Egg_cast

## Mite/egg model best for quantification from mite brushing samples

[Link to model hosted on Roboflow](https://universe.roboflow.com/gent-lab/tssm-detection-v2/model/27)

Two classes:
* Mite
* Viable_egg

