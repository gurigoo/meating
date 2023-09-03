from fastapi import FastAPI, UploadFile
import uvicorn
import torch

from models import resnet
from data import dataset, pre_process

app= FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet.resnet101

@app.on_event('startup')
def start_app():
    model.load_state_dict(torch.load('./log/03/00027.pt', map_location=device))
    model.to(device)
    model.eval()
    print('init app')

@app.on_event('shutdown')
def ends_app():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('shutdown app')

@app.post("/inference")
async def inference(img):
    print('inference start')
    img = pre_process.resizing(img['file'],480)
    img = dataset.val_transform(img)
    model.eval()
    with torch.no_grad():
        img.to(device)
        output = model(img)[0]
        cls = torch.argmax(output)
    score = int((4-cls)*25)
    print(score)
    return score




if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)