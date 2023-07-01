
from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.utils.types import compose_argument_block
from dlhub_sdk import DLHubClient
import json
import pickle as pkl
from app import inference


# Read in model from disk
model_info = PythonStaticMethodModel.from_function_pointer(inference)

# Define the name and title for the model

model_info.set_title("Ziatdinov AtomNet")
model_info.set_name("Ziatdinov_AtomNet")

# Verify authors and affiliations
model_info.set_creators(["Maxim, Ziatdinov", "Wei, Jingrui"], [["Oak Ridge National Laboratory"], ["University of Wisconsin Madison"]])

# Describe the scientific purpose of the model
model_info.set_domains(["general", "materials science", "microscopy"])
# model_info.set_abstract("A model for atom localization in atomically-resolved STEM images. An encoder–decoder type U-net architectured CNN network is used. It takes a numpy array of image data as input and give atom column coordinates as output. Two set of model weights, image resized ratio are available. ")

# Add references for the model
# model_info.add_related_identifier("10.1021/acsnano.7b07504", "DOI", "IsDescribedBy")  # Example: Paper describing the model
# model_info.add_alternate_identifier("https://github.com/ziatdinovmax/AICrystallographer/tree/master/AtomNet", "URL")  # Example: Github documenting the model

model_info.add_file('requirements.txt')

# Describe the inputs and outputs of the model
model_info.set_inputs('dict', 'Dict of input image array and resize factor, or a list of dicts',
                       properties = {
                         "image": {"type": 'ndarray', "shape": ([None,None])},
                         "change_size": {"type": 'float'}, 
                         "modelweights": {"type": 'string'}
                         })
model_info.set_outputs('ndarray', 'List of the coordinates of located atomic columns for all the inputs')

# Add the file describing the python class method, and associated files of the pytorch nets
# if submit net files and model weights files in the root dir of the folder, no need to add_directory?
model_info.add_file('app.py')
#model_info.add_file('atomfind.py')
#model_info.add_file('dcnn.py')
#model_info.add_file('nnblocks.py')
#model_info.add_file('utils.py')
model_info.add_file('G-Si-DFT0-1-4-best_weights.pt')
model_info.add_file('cubic-best_weights.pt')


# Check the schema against a DLHub Schema
validate_against_dlhub_schema(model_info.to_dict(), 'servable')

# # Save the metadata
# with open('dlhub.json', 'w') as fp:
#       # this was Jingrui's original line, dlhub-sdk isnt set up like that anymore
#     # json.dump(model_info.to_dict(save_class_data=True), fp, indent=2)
#     json.dump(model_info.to_dict(), fp, indent=2)

# publish the model
client = DLHubClient()
taskid = client.publish_servable(model_info)
print(taskid)
