# Click-to-Move - Guia de Execução

Este guia organiza a inicialização do LiDAR/ODrive, Navigation2, Gateway IS-ROS2, Dynamic Follower e o cliente Click-to-Go.

## Pré-requisitos
- Docker instalado e funcionando.
- Acesso a USB (containers com --privileged e -v /dev/bus/usb:/dev/bus/usb).
- Arquivo de mapa disponível (map_cam_odom_exp_170601.yaml).
- Ajustar a variável USER no script nav2-rviz2-container.sh para o usuário da sua máquina.
- Rede com acesso aos brokers AMQP usados no projeto.

Dica: Você usará múltiplos terminais (pelo menos 5). Mantenha-os nomeados por etapa.

---

## Passo 1 (Conectado no Robô): Inicializar LiDAR e ODrive (is-robis-ros2)
1) Subir o container
```
sudo docker run --rm --privileged -it --network=host -v "$(pwd):/workspace/src" -v /dev/bus/usb:/dev/bus/usb --name=robis_ro2 matheusdutra0207/is-robis-ros2:0.0.4 bash
```

2) Iniciar o node do ODrive
```
source install/setup.bash
ros2 launch odrive_ros2_pkg is_robis_ros2_launch.py publish_odom_tf:=true
```

---

## Passo 2: Navigation2 (Na sua máquina)

### Terminal 1 (container c/ RViz e Nav2)
1) Clone o repositório necessário (navigation2) no seu $HOME e acesse
```
git clone https://github.com/labvisio/navigation2.git
cd navigation2
```

2) No arquivo nav2-rviz2-container.sh, atualize a variável USER para o usuário da sua máquina.

3) Suba o container utilitário
```
sudo ./nav2-rviz2-container.sh
```

4) Fonte do ROS
```
source /opt/ros/humble/setup.bash
```

5) Vá até a pasta de mapas e rode a localização
```
cd /opt/ros/humble/share/nav2_bringup/maps/
ros2 launch nav2_bringup localization_launch.py map:=map_cam_odom_exp_170601.yaml
```

### Terminal 2 (dentro do container ros-utilities)
1) Entrar no container
```
sudo docker exec -it ros-utilities bash
```

2) Fonte do ROS
```
source /opt/ros/humble/setup.bash
```

3) Re-inicializar a localização global
```
ros2 service call /reinitialize_global_localization std_srvs/Empty
```

4) Abrir RViz2
```
rviz2
```
Lembre-se de carregar a config do RViz “Fuse”.

### Terminal 3 (dentro do container ros-utilities)
1) Entrar no container
```
sudo docker exec -it ros-utilities bash
```

2) Fonte do ROS
```
source /opt/ros/humble/setup.bash
```

3) Iniciar o Navigation2
```
ros2 launch nav2_bringup navigation_launch.py
```

---

## Passo 3: Inicializar o Gateway IS-ROS2
### Terminal 4
1) Subir o container
```
sudo docker run --rm -it --network=host matheusdutra0207/is-ros2-gateway:0.0.1 bash
```

2) Fonte do workspace
```
source install/setup.bash
```

3) Rodar o gateway (ajuste o URI conforme sua rede)
```
ros2 run is_ros2_gateway is_ros2_gateway --uri amqp://10.20.5.2:30000
```

---

## Passo 4: Ativar o Node Dynamic Follower (Na sua máquina)
### Terminal 5
Na sua pasta de workspaces ROS2 (ex.: ~/ros2_ws)
```
colcon build
source install/setup.bash
ros2 run dynamic_follower follower_node
```

---

## Passo 5: Rodar o cliente Click-to-Go
### Terminal 6
1) Execute o cliente indicando as câmeras desejadas (ex.: 0 1 2 3)
```
cd click-to-move
pip install -r requirements.txt
cd src
python3 client.py -c 0 1 2 3
```
Observação: o cliente usa o broker_uri definido em ../conf/config.json.

2) Controles rápidos do cliente:
- q: sair (envia stop)
- c: limpar overlays

Clique na imagem para enviar o alvo ao robô (coordenada transformada para o mundo).

---

## Dicas e Solução de Problemas
- RViz: Lembre-se de carregar a configuração “Fuse”.
- Caminho do ROS: use sempre `source /opt/ros/humble/setup.bash` (observe a barra inicial).
- Mapas: verifique se o arquivo YAML está disponível em `/opt/ros/humble/share/nav2_bringup/maps/` no container.
- USB/ODrive: use o container com `--privileged` e o bind de `/dev/bus/usb`.
- Rede: verifique URIs AMQP e conectividade (porta/host corretos).
- Client: o broker_uri do config.json é usado para publicar em `ros.skeleton_pose`.