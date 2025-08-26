conda create --name captioning python=3.12
conda activate captioning
echo "Conda environment 'captioning' created and activated."
echo ""

pip install -r requirements.txt
echo "Required packages installed."

mkdir -p data/source data/captions data/demo data/cache

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "Kaggle API token not found."
    echo "Place your Kaggle API token in ~/.kaggle/kaggle.json"
fi;

echo ""
echo "Kaggle API token found. Downloading dataset..."
cd data
curl -L -o unsplash-random-images-collection.zip\
  https://www.kaggle.com/api/v1/datasets/download/lprdosmil/unsplash-random-images-collection

echo ""
unzip unsplash-random-images-collection.zip
mv unsplash-images-collection source
rm unsplash-random-images-collection.zip

echo ""
echo "Dataset downloaded and extracted. Check data/source for images."

echo ""
echo "Pulling Ollama model qwen2.5vl:7b"
ollama pull qwen2.5vl:7b
echo "Ollama model qwen2.5vl:7b pulled."
