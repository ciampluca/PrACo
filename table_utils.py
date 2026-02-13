
MODEL_NAMES_TO_LATEX_DICT = {
    "CounTX" : "CounTX~\cite{AminiNaieni23} {\\footnotesize (BMVC '23)}",
    "CLIP-Count" : "CLIP-Count~\cite{10.1145/3581783.3611789} {\\footnotesize (ACM MM '23)}",
    "VLCounter" : "VLCounter~\cite{Kang_Moon_Kim_Heo_2024} {\\footnotesize (AAAI '24)}",
    "TFPOC" : "TFPOC~\cite{10483595} {\\footnotesize (WACV '24)}",
    "DAVE" : "DAVE~\cite{Pelhan_2024_CVPR} {\\footnotesize (CVPR '24)}",
    "ZSC" : "ZSC~\cite{10204688} {\\footnotesize (CVPR '23)}",
    "PseCo" : "PseCo~\cite{DBLP:conf/cvpr/HuangD0ZS24} {\\footnotesize (CVPR '24)}",
    "GroundingREC" : "GroundingREC~\cite{10656642} {\\footnotesize (CVPR '24)}$^*$",
    "GroundingRECFSC" : "GroundingREC~\cite{10656642} {\\footnotesize (CVPR '24)}$^\$$",
    "CountGD" : "CountGD~\cite{DBLP:journals/corr/abs-2407-04619} {\\footnotesize (NeurIPS '24)}",
    "FixedPointPromptCounting" : "UPC~\cite{Lin_Chan_2024} {\\footnotesize (AAAI '24)}"
}

def model_name_to_table_model_name(model_name):
    if model_name in MODEL_NAMES_TO_LATEX_DICT.keys():
        return MODEL_NAMES_TO_LATEX_DICT[model_name]
    return model_name

models_ordering = ["ZSC", "CounTX", "CLIP-Count", "VLCounter", "TFPOC", "DAVE", "PseCo", "GroundingREC", "GroundingRECFSC", "UPC", "FixedPointPromptCounting", "CountGD"]

def get_ordered_models_list():
    tmp = models_ordering.copy()
    tmp.remove("UPC")
    return tmp

models_colors = {
    "ZSC" : "blue",
    "CounTX" : "orange",
    "CLIP-Count" : "green",
    "VLCounter" : "red",
    "TFPOC" : "purple",
    "DAVE" : "brown",
    "PseCo" : "pink",
    "GroundingREC" : "gray",
    "GroundingRECFSC" : "gray",
    "CountGD" : "magenta",
    "UPC" : "yellow",
    "FixedPointPromptCounting" : "yellow"
}

def get_model_color(model_name):
    if model_name in models_colors.keys():
        return models_colors[model_name]
    return "black"

def get_shortened_name(model_name):
    if model_name in MODEL_NAMES_TO_LATEX_DICT.keys():
        return MODEL_NAMES_TO_LATEX_DICT[model_name].split("~")[0]
    return model_name