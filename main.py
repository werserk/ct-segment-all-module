from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.model import init_model
from src.production import make_prediction
from io import BytesIO
import nibabel as nib

app = FastAPI()

model = init_model("data/model.pt")


@app.post("/segmentation/")
async def segmentation(file: UploadFile = File(...)):
    contents = await file.read()
    file_stream = BytesIO(contents)
    img_nifti = nib.load(file_stream)
    mask = make_prediction(model, img_nifti)  # 104 classes, it needs additional filters (like in Playground.ipynb)
    return JSONResponse(content={"mask": mask.tolist()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
