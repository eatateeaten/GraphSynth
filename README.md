This project is going to cover a no-code user-friendly Visual/graph-compiler for all existing neuralnet architectures.
As well as graph representations that translate to PyTorch, JAX, Tensorflow model and training routine code. 
I will also develop a distributed training/inferencing framework based on this project. 

# Installation

Install npm packages:

```
npm install
```

Install pip packages for the runtime:

```
cd runtime
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```

# Running

Start the Jupyter runtime:

```
jupyter notebook --ServerApp.allow_origin="http://localhost:5173" --ServerApp.allow_credentials=True --no-browser
```

Start the Electron app:

```
npm run dev
```

You can paste the Jupyter link into the connection prompt to be able to execute generated code.
