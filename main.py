from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import torch

from models import resnet
from data import dataset, pre_process

app= FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet.resnetregression101

@app.on_event('startup')
def start_app():
    model.load_state_dict(torch.load('./log/05/00006.pt', map_location=device))
    model.to(device)
    model.eval()
    print('init app')

@app.on_event('shutdown')
def ends_app():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('shutdown app')

@app.post("/inference/")
async def inference(img: bytes = File()):
    print('inference start')
    img = pre_process.resizing(img,480)
    img = dataset.val_transform(img)
    img = img.unsqueeze(0)
    print(img.shape)
    model.eval()
    with torch.no_grad():
        img.to(device)
        output = model(img)[0]
        print(output)
        score = min(int(output*100),100)
    print(score)
    return score




if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)