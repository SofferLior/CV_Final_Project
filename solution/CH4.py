from xcpetion import build_xception_backbone
from utils import get_nof_params

model = build_xception_backbone()
sum_param = get_nof_params(model)
print(sum_param)