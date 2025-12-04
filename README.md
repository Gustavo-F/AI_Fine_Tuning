# Guia básico de configuração e utilização

## Ambiente e dependências

### Criar abiente `venv` necessário para a execução do projeto
```bash
python3 -m venv .venv
```

### Ativação do ambiente
```bash
source .venv/bin/activate
```

### Instalação das dependências
````bash
pip install -r requirements.txt
````


## Execução dos scripts

### Executar script de treinamento
```bash
python training/train.py
```

### Realizar merge do output do script de treinamento com o modelo pré-treinado
```bash
python training/merge_lora.py
```

## Converter modelo para GGUF e Quantização
Para essa etapa há a necessiade de utilizar o `llama.cpp` que irá realizar os processo de converção de quantização. Faça o clone do repositório [aqui](https://github.com/ggml-org/llama.cpp.git) e realize o build do mesmo conforme o [passo a passo](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

### Converção para GGUF
```bash
python llama.cpp/convert_hf_to_gguf.py ./merged_model/ --outtype f16 --outfile your_treined_model.gguf
```

### Quantização do Modelo
OBS: esse processo só é possível após realizar o build do `llama.cpp`.
```bash
./llama.cpp/build/bin/llama-quantize your_treined_model.gguf your_custom_model.Q4_K_M.gguf Q4_K_M
```