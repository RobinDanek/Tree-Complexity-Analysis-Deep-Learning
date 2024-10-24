import numpy as np

AVG_NUM = 213996
cloudLabelType_avg = [('point_cloud', float, (AVG_NUM,3)), ('label', float)]

C100K = 10**5
cloudLabelType_100k= [('point_cloud', float, (C100K,3)), ('label', float)]

C10K = 10**4
cloudLabelType_10k= [('point_cloud', float, (C10K,3)), ('label', float)]

predictionType = [('tree_id', 'U30'), ('label', float), ('label_pred', float)]