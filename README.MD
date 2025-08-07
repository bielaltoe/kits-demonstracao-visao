# Kit de Demonstração de Visão Computacional

Este kit contém demonstrações interativas que mostram o poder da visão computacional com Python. As demonstrações são projetadas para serem divertidas e educativas.

## Requisitos do Sistema

- Computador com Ubuntu Linux (ou outra distribuição baseada em Debian)
- Webcam
- Python 3.6 ou superior

## Guia de Instalação

Siga estes passos para configurar e executar as demonstrações. Abra um terminal e digite os comandos abaixo:

### Passo 1: Instalar as dependências do sistema

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev portaudio19-dev
```

### Passo 2: Clonar ou baixar este repositório

Se você baixou um arquivo ZIP, extraia-o para uma pasta. 
Se você está usando Git, você pode clonar o repositório:

```bash
git clone [URL_DO_REPOSITORIO]
cd kits-demonstracao-visao
```

### Passo 3: Instalar as bibliotecas Python necessárias

```bash
sudo apt-get install portaudio19-dev
pip install -r requirements.txt
```

## Demonstrações Disponíveis

### 1. Contador de Dedos

Esta demonstração detecta sua mão e conta quantos dedos você está mostrando.

**Para executar:**
```bash
python3 finger_count.py
```

**Como usar:**
- Mostre sua mão para a câmera
- O programa contará automaticamente quantos dedos estão levantados
- Pressione 'q' para sair

### 2. Desenho com Marcadores ArUco

Esta demonstração permite desenhar na tela usando marcadores ArUco (códigos QR especiais).

**Para executar:**
```bash
python3 aruco_draw.py
```

**Como usar:**
- Imprima os marcadores ArUco (encontrados na pasta "markers")
- Mostre os marcadores para a câmera
- Mova os marcadores para desenhar
- Pressione 'q' para sair


### 3. Theremin Virtual

Esta demonstração cria um instrumento musical controlado pelos movimentos das mãos.

**Para executar:**
```bash
cd ..
python3 theremin.py
```

**Como usar:**
- Use a mão esquerda para controlar:
  - Posição horizontal (X): controla o tom (esquerda = mais grave, direita = mais agudo)
  - Posição vertical (Y): controla o volume (mais alto = mais alto, mais baixo = mais baixo)
- Use a mão direita para gestos:
  - Gesto de paz (V): muda o tipo de som
  - Gesto de pinça: produz um efeito percussivo
- Pressione 'q' para sair

## Solução de Problemas

Se você encontrar erros durante a instalação ou execução:

1. **Erro com a webcam**: Verifique se sua webcam está conectada e funcionando
   ```bash
   ls -l /dev/video*
   ```

2. **Erro com bibliotecas**: Tente reinstalar as dependências
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Erros com PyAudio**: Execute o comando específico para instalar o PyAudio
   ```bash
   sudo apt-get install portaudio19-dev && pip install pyaudio
   ```

4. **Problemas de desempenho**: Feche outros programas que possam estar usando a webcam ou consumindo muita CPU