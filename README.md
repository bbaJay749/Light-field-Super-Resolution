# Light-field-Super-Resolution
Light field data Super Resolution, both Angular &amp; Spatial \n

VDSR_model.py - to train single image spatial SR network, VDSR.\n
VDSR_gen_SR_image.py - to SR each images of the light field data.\n
\n
Angular_dataset_gen.py - to generate data from a light-field datatset.\n
Angular_model.py - to train angular SR network. gets only 2 inputs. \n
Angular_test.py - to SR light field data array. (X2)\n
\n
\n
VDSR network (Sptial SR) works well but Angualr SR network should be modified more. \n
It works by getting only 2 images and generates a middle-view image but I assume it would be better \n
by applying some structure that extracts all the features and relationships among each images of the LF image array.\n
