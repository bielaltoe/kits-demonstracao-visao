#!/usr/bin/env python3
# filepath: /home/gabriel/Documents/codes/projeto_extencionista/theremin.py
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
import pyaudio
import wave
from collections import deque

# --- AudioEngine Class ---
class AudioEngine:
    """
    Responsável pela geração de áudio em tempo real.
    Implementa diferentes formas de onda e controles de volume/frequência.
    """
    def __init__(self):
        # Configurações de áudio
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.channels = 1
        
        # Estado do áudio
        self.is_playing = False
        self.volume = 0.0
        self.target_volume = 0.0
        self.frequency = 440.0
        self.target_frequency = 440.0
        self.waveform = 'sine'  # 'sine', 'square', 'triangle', 'sawtooth', 'synth'
        
        # Configurações de transição
        self.frequency_smoothing = 0.1  # Taxa de transição de frequência
        self.volume_smoothing = 0.1     # Taxa de transição de volume
        
        # Parâmetros de sintetizador
        self.lfo_rate = 5.0        # Taxa do LFO (Low Frequency Oscillator) em Hz
        self.lfo_depth = 0.1       # Profundidade do LFO (0.0-1.0)
        self.filter_cutoff = 0.8   # Frequência de corte do filtro (0.0-1.0)
        self.resonance = 0.2       # Ressonância do filtro (0.0-1.0)
        self.detune = 0.02         # Desafinação para efeito "fat sound" (0.0-0.1)
        
        # Proteção para acesso threadsafe
        self.lock = threading.Lock()
        
        # Inicializar o objeto PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Buffers de percussão
        self.percussion_samples = None
        self.percussion_index = 0
        self.percussion_active = False
        self.percussion_volume = 8.0  # Aumentado para 8.0 para garantir que o som grave seja bem audível
        
        # Tentar carregar sons de percussão ou gerar um som sintético como fallback
        self._init_percussion()
        
    def _init_percussion(self):
        """Inicializa o som de percussão (um som curto que será tocado no gesto de pinça)"""
        # Criar um som estilo 808 (grave com decaimento longo)
        duration = 0.5  # Aumentado para 500ms para um som mais longo e grave
        num_samples = int(self.sample_rate * duration)
        self.percussion_samples = np.zeros(num_samples, dtype=np.float32)
        
        # Envelope ADSR para som estilo 808
        attack = int(0.005 * self.sample_rate)  # 5ms - ataque rápido
        decay = int(0.1 * self.sample_rate)     # 100ms - decay moderado
        sustain_level = 0.7                     # Nível de sustain alto para mais presença
        release = int(0.4 * self.sample_rate)   # Release longo para som grave
        
        # Phase 1: Attack - abrupto para definição do som
        self.percussion_samples[:attack] = np.linspace(0, 1.0, attack)
        
        # Phase 2: Decay até o nível de sustain
        decay_end = attack + decay
        decay_curve = np.linspace(1.0, sustain_level, decay)
        self.percussion_samples[attack:decay_end] = decay_curve
        
        # Phase 3: Sustain & Release - decaimento exponencial lento
        release_start = decay_end
        t = np.linspace(0, 1, num_samples - release_start)
        release_curve = sustain_level * np.exp(-3 * t)  # Decaimento mais lento (-3 em vez de -8)
        self.percussion_samples[release_start:] = release_curve
        
        # Adicionar componentes de frequência para som tipo 808
        # Frequência fundamental grave (típica de 808)
        bass_freq = 60  # Hz - frequência muito grave tipo 808
        t_full = np.arange(num_samples) / self.sample_rate
        
        # Senoide principal com pitch bend para baixo (característica do 808)
        pitch_bend = np.linspace(1.2, 1.0, num_samples)  # Começa ligeiramente mais agudo e desce
        bass_component = np.sin(2 * np.pi * bass_freq * pitch_bend * t_full)
        
        # Adicionar um pouco de distorção para dar corpo ao som
        bass_component = np.tanh(1.5 * bass_component)
        
        # Adicionar harmônicos para dar definição ao som
        harmonic1 = 0.3 * np.sin(2 * np.pi * bass_freq * 2 * t_full) * np.exp(-10 * t_full)
        harmonic2 = 0.15 * np.sin(2 * np.pi * bass_freq * 3 * t_full) * np.exp(-15 * t_full)
        
        # Combinar componentes com o envelope
        self.percussion_samples = self.percussion_samples * (bass_component + harmonic1 + harmonic2)
        
        # Normalizar para evitar clipping
        self.percussion_samples = self.percussion_samples / np.max(np.abs(self.percussion_samples))
        
    def start(self):
        """Inicia o stream de áudio"""
        if self.stream is not None:
            return
            
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._audio_callback
        )
        self.is_playing = True
        
    def stop(self):
        """Para o stream de áudio"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        
    def set_frequency(self, freq):
        """Define a frequência alvo (com transição suave)"""
        with self.lock:
            # Limitar frequências para evitar sons desagradáveis
            freq = max(55.0, min(freq, 1760.0))  # Entre A1 e A6
            self.target_frequency = freq
            
    def set_volume(self, vol):
        """Define o volume alvo (com transição suave)"""
        with self.lock:
            self.target_volume = max(0.0, min(vol, 1.0))
            
    def set_waveform(self, waveform):
        """Altera o tipo de forma de onda"""
        if waveform in ('sine', 'square', 'triangle', 'sawtooth', 'synth'):
            with self.lock:
                self.waveform = waveform
                
    def trigger_percussion(self):
        """Aciona o som de percussão"""
        with self.lock:
            self.percussion_active = True
            self.percussion_index = 0
            
    def cleanup(self):
        """Limpa recursos do PyAudio"""
        self.stop()
        self.p.terminate()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback chamado quando o sistema de áudio precisa de mais amostras"""
        with self.lock:
            # Interpolação suave para frequência e volume
            self.frequency += (self.target_frequency - self.frequency) * self.frequency_smoothing
            self.volume += (self.target_volume - self.volume) * self.volume_smoothing
            
            # Gerar a fase para este buffer
            t = np.arange(frame_count) / self.sample_rate
            phase = 2 * np.pi * self.frequency * t
            
            # Componentes para efeitos de sintetizador
            lfo = self.lfo_depth * np.sin(2 * np.pi * self.lfo_rate * t)  # LFO para modulação
            
            # Escolher a função para a forma de onda
            if self.waveform == 'sine':
                samples = self.volume * np.sin(phase)
            elif self.waveform == 'square':
                samples = self.volume * np.sign(np.sin(phase))
            elif self.waveform == 'triangle':
                samples = self.volume * (2 / np.pi) * np.arcsin(np.sin(phase))
            elif self.waveform == 'sawtooth':
                samples = self.volume * (2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi))))
            elif self.waveform == 'synth':
                # synth: Som estilo FM (Frequency Modulation)
                # Modulador
                mod_freq = self.frequency * 2  # Modulador com frequência 2x
                mod_phase = 2 * np.pi * mod_freq * t
                mod_signal = np.sin(mod_phase) * 5.0  # Índice de modulação
                # Portadora modulada em frequência
                samples = self.volume * np.sin(phase + mod_signal)
                # Adicionar um pouco de modulação de amplitude com LFO
                samples = samples * (1.0 + lfo * 0.3)
            else:
                samples = self.volume * np.sin(phase)  # Default to sine
                
            # Adicionar percussão se ativa - com prioridade sobre o som de fundo
            if self.percussion_active and self.percussion_samples is not None:
                remaining = len(self.percussion_samples) - self.percussion_index
                
                if remaining > 0:
                    n = min(remaining, frame_count)
                    
                    # Reduzir temporariamente o volume do som principal durante a percussão
                    # para deixar a percussão se destacar
                    samples_vol_reduced = samples.copy()
                    volume_reduction = np.linspace(0.3, 1.0, n)  # Gradualmente volta ao volume normal
                    samples_vol_reduced[:n] = samples[:n] * volume_reduction
                    
                    # Adicionar a percussão com volume aumentado
                    samples_vol_reduced[:n] += self.percussion_volume * self.percussion_samples[self.percussion_index:self.percussion_index+n]
                    samples = samples_vol_reduced
                    
                    self.percussion_index += n
                    
                    if self.percussion_index >= len(self.percussion_samples):
                        self.percussion_active = False
                else:
                    self.percussion_active = False
            
            # Converter para float32 e formatar adequadamente
            output = samples.astype(np.float32)
            
        return (output, pyaudio.paContinue)


# --- GestureInterpreter Class ---
class GestureInterpreter:
    """
    Interpreta os dados das mãos do MediaPipe e os traduz em comandos musicais.
    """
    def __init__(self):
        # Configurações
        self.frequency_min = 110.0   # A2
        self.frequency_max = 880.0   # A5
        
        # Máquina de estados para gestos
        self.pinch_state = "IDLE"  # IDLE, DETECTED, ACTIVE
        self.pinch_frames = 0
        self.pinch_threshold = 0.035  # Aumentado para ser mais tolerante com o gesto de pinça
        self.pinch_cooldown = 0.1    # Reduzido para permitir percussões mais rápidas
        self.last_pinch_time = 0
        
        # Estado do interpretador
        self.current_waveform = 'sine'
        self.waveform_options = ['sine', 'square', 'triangle', 'sawtooth', 'synth']
        self.waveform_index = 0
        self.waveform_cooldown = 2.0  # 2 segundos de cooldown para mudança de forma de onda
        self.last_waveform_change_time = 0
        
    def get_distance(self, p1, p2):
        """Calcula a distância euclidiana entre dois pontos (landmarks)"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
        
    def map_value(self, value, in_min, in_max, out_min, out_max):
        """Mapeia um valor de um intervalo para outro"""
        # Limitar valor ao intervalo de entrada
        value = max(in_min, min(in_max, value))
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    def count_fingers_up(self, hand_landmarks):
        """Conta dedos levantados de forma robusta"""
        if not hand_landmarks:
            return 0
        
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        count = 0

        # Polegar (verifica a posição X em relação à palma)
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
             count += 1

        # Outros 4 dedos (verifica a posição Y)
        for tip_index in finger_tips:
            if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:
                count += 1
                
        return count
        
    def is_peace_sign(self, hand_landmarks):
        """Verifica se o gesto é um sinal de paz (V)"""
        if not hand_landmarks:
            return False
            
        # Verificar se os dedos indicador e médio estão levantados
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
        
        # Verificar se os outros dedos estão abaixados
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
        
        return index_up and middle_up and ring_down and pinky_down
        
    def check_pinch(self, hand_landmarks, audio_engine):
        """Verifica e processa o gesto de pinça para tocar percussão"""
        if not hand_landmarks:
            return
            
        # Calcular distância entre o polegar e o indicador
        pinch_distance = self.get_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
        
        # Adicionando feedback visual ao console para debug
        if pinch_distance < self.pinch_threshold:
            print(f"Pinch detected! Distance: {pinch_distance:.4f}")
        
        # Máquina de estados para o gesto de pinça
        if self.pinch_state == "IDLE":
            if pinch_distance < self.pinch_threshold:
                self.pinch_state = "DETECTED"
                self.pinch_frames = 1
        
        elif self.pinch_state == "DETECTED":
            if pinch_distance < self.pinch_threshold:
                self.pinch_frames += 1
                if self.pinch_frames >= 2:  # Reduzido de 3 para 2 frames para resposta mais rápida
                    current_time = time.time()
                    if current_time - self.last_pinch_time > self.pinch_cooldown:
                        self.pinch_state = "ACTIVE"
                        self.last_pinch_time = current_time
                        # Ação de pinça: tocar percussão
                        audio_engine.trigger_percussion()
                        print("PERCUSSION TRIGGERED!")  # Debug info
                        return True
            else:
                self.pinch_state = "IDLE"
                
        elif self.pinch_state == "ACTIVE":
            if pinch_distance >= self.pinch_threshold:
                self.pinch_state = "IDLE"
                
        return False
        
    def check_waveform_change(self, hand_landmarks, audio_engine):
        """Verifica e processa o gesto de 'paz' para trocar a forma de onda"""
        if not hand_landmarks:
            return False
            
        if self.is_peace_sign(hand_landmarks):
            current_time = time.time()
            if current_time - self.last_waveform_change_time > self.waveform_cooldown:
                self.last_waveform_change_time = current_time
                
                # Trocar de forma de onda
                self.waveform_index = (self.waveform_index + 1) % len(self.waveform_options)
                self.current_waveform = self.waveform_options[self.waveform_index]
                audio_engine.set_waveform(self.current_waveform)
                return True
                
        return False
        
    def process_left_hand(self, hand_landmarks, width, height, audio_engine):
        """Processa a mão esquerda para controlar frequência e volume"""
        if not hand_landmarks:
            return
            
        # Usar a posição do indicador (landmark 8) como referência
        x_position = hand_landmarks.landmark[8].x
        y_position = hand_landmarks.landmark[8].y
        
        # Mapear X para frequência (escala logarítmica)
        frequency = self.frequency_min * (self.frequency_max / self.frequency_min) ** x_position
        audio_engine.set_frequency(frequency)
        
        # Mapear Y para volume (invertendo, pois Y aumenta para baixo)
        volume = self.map_value(y_position, 0.1, 0.9, 1.0, 0.0)
        audio_engine.set_volume(volume)
        
        return frequency, volume
        
    def process_right_hand(self, hand_landmarks, audio_engine):
        """Processa a mão direita para gestos de controle"""
        if not hand_landmarks:
            return
            
        # Verificar gesto de pinça para percussão
        pinch_triggered = self.check_pinch(hand_landmarks, audio_engine)
        
        # Verificar gesto de paz para mudar forma de onda
        waveform_changed = self.check_waveform_change(hand_landmarks, audio_engine)
        
        return {
            'pinch_triggered': pinch_triggered,
            'waveform_changed': waveform_changed,
            'current_waveform': self.current_waveform
        }


# --- Visualizer Class ---
class Visualizer:
    """
    Responsável pela parte visual da aplicação, incluindo feedback ao usuário.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.max_particles = 100
        self.waveform_colors = {
            'sine': (0, 255, 255),      # Ciano
            'square': (255, 0, 255),    # Magenta
            'triangle': (255, 255, 0),  # Amarelo
            'sawtooth': (0, 255, 0),    # Verde
            'synth': (255, 128, 0)     # Laranja
        }
        
        # Área do visualizador de forma de onda
        self.wave_vis_x = 50
        self.wave_vis_y = 50
        self.wave_vis_width = 200
        self.wave_vis_height = 100
        
    def add_particles(self, x, y, color, count=10):
        """Adiciona partículas ao sistema"""
        for _ in range(count):
            # Velocidade aleatória
            vx = np.random.uniform(-2, 2)
            vy = np.random.uniform(-4, -1)  # Sempre sobe
            # Tempo de vida aleatório
            lifetime = np.random.uniform(0.5, 1.5)
            # Tamanho aleatório
            size = np.random.uniform(2, 6)
            
            self.particles.append({
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'size': size
            })
            
    def update_particles(self):
        """Atualiza o sistema de partículas"""
        new_particles = []
        
        for p in self.particles:
            # Atualizar posição
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Aplicar gravidade
            p['vy'] += 0.1
            
            # Diminuir tempo de vida
            p['lifetime'] -= 0.016  # ~60fps
            
            # Manter partículas com vida restante
            if p['lifetime'] > 0:
                new_particles.append(p)
                
        # Limitar número de partículas
        self.particles = new_particles[-self.max_particles:] if len(new_particles) > self.max_particles else new_particles
        
    def draw_particles(self, img):
        """Desenha o sistema de partículas"""
        for p in self.particles:
            # Calcular a opacidade baseada no tempo de vida
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            # Calcular cor com alpha
            color_with_alpha = (
                int(p['color'][0] * alpha / 255),
                int(p['color'][1] * alpha / 255),
                int(p['color'][2] * alpha / 255)
            )
            # Desenhar partícula
            cv2.circle(img, (int(p['x']), int(p['y'])), int(p['size']), color_with_alpha, -1)
            
    def draw_waveform(self, img, waveform, frequency, volume):
        """Desenha uma visualização da forma de onda atual"""
        # Desenhar fundo
        cv2.rectangle(img, 
            (self.wave_vis_x, self.wave_vis_y), 
            (self.wave_vis_x + self.wave_vis_width, self.wave_vis_y + self.wave_vis_height), 
            (30, 30, 30), -1)
        cv2.rectangle(img, 
            (self.wave_vis_x, self.wave_vis_y), 
            (self.wave_vis_x + self.wave_vis_width, self.wave_vis_y + self.wave_vis_height), 
            (200, 200, 200), 1)
            
        # Desenhar forma de onda
        color = self.waveform_colors.get(waveform, (255, 255, 255))
        x_vals = np.linspace(0, 2*np.pi, self.wave_vis_width)
        
        points = []
        mid_y = self.wave_vis_y + self.wave_vis_height // 2
        
        for i, x in enumerate(x_vals):
            if waveform == 'sine':
                y = volume * np.sin(x) * (self.wave_vis_height * 0.4) + mid_y
            elif waveform == 'square':
                y = volume * np.sign(np.sin(x)) * (self.wave_vis_height * 0.4) + mid_y
            elif waveform == 'triangle':
                y = volume * (2 / np.pi) * np.arcsin(np.sin(x)) * (self.wave_vis_height * 0.4) + mid_y
            elif waveform == 'sawtooth':
                phase = x % (2*np.pi)
                y = volume * (2 * (phase / (2*np.pi) - 0.5)) * (self.wave_vis_height * 0.4) + mid_y
            elif waveform == 'synth':
                # Visualização modulada para synth
                mod = np.sin(x * 2) * 1.0
                y = volume * np.sin(x + mod) * (self.wave_vis_height * 0.4) + mid_y
            else:
                y = mid_y
                
            points.append((self.wave_vis_x + i, int(y)))
            
        # Desenhar linhas conectando os pontos
        for i in range(1, len(points)):
            cv2.line(img, points[i-1], points[i], color, 2)
            
        # Mostrar informações de frequência e volume
        freq_text = f"Freq: {int(frequency)} Hz"
        vol_text = f"Vol: {int(volume*100)}%"
        cv2.putText(img, freq_text, 
            (self.wave_vis_x, self.wave_vis_y + self.wave_vis_height + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, vol_text, 
            (self.wave_vis_x + self.wave_vis_width - 80, self.wave_vis_y + self.wave_vis_height + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Mostrar nome da forma de onda
        cv2.putText(img, waveform.upper(), 
            (self.wave_vis_x + self.wave_vis_width // 2 - 30, self.wave_vis_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
    def draw_hand_tracking(self, img, cursor_hand_landmarks, action_hand_landmarks, mp_hands, mp_draw):
        """Desenha as marcações das mãos"""
        if cursor_hand_landmarks:
            mp_draw.draw_landmarks(img, cursor_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 128, 128), thickness=2))
                
        if action_hand_landmarks:
            mp_draw.draw_landmarks(img, action_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(128, 0, 128), thickness=2))
                
    def draw_control_points(self, img, cursor_hand_pts, action_hand_pts, frequency, volume, waveform):
        """Desenha pontos de controle e informações visuais adicionais"""
        # Desenhar ponto de controle da mão esquerda (frequência/volume)
        if cursor_hand_pts:
            control_point = cursor_hand_pts[8]  # Ponta do indicador
            color = self.waveform_colors.get(waveform, (255, 255, 255))
            
            # Círculos concêntricos com tamanho baseado no volume
            radius = int(30 * volume) + 10
            cv2.circle(img, control_point, radius, color, 2)
            cv2.circle(img, control_point, radius // 2, color, 1)
            
            # Linha para o centro
            cv2.line(img, control_point, 
                     (control_point[0], control_point[1] - radius), 
                     color, 1)
            cv2.line(img, control_point, 
                     (control_point[0] + radius, control_point[1]), 
                     color, 1)
            
            # Adicionar partículas
            self.add_particles(control_point[0], control_point[1], color, count=1)
            
        # Desenhar informações da mão direita (gestos)
        if action_hand_pts:
            # Verificar aproximação entre polegar e indicador para feedback visual de pinça
            thumb_tip = action_hand_pts[4]
            index_tip = action_hand_pts[8]
            
            # Calcular distância entre as pontas dos dedos
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
            
            # Se os dedos estão próximos, desenhar feedback visual para pinça
            if distance < 50:  # Valor em pixels, ajuste conforme necessário
                # Desenhar linha conectando os dedos
                cv2.line(img, thumb_tip, index_tip, (0, 0, 255), 2)
                
                # Desenhar círculos nas pontas dos dedos
                cv2.circle(img, thumb_tip, 10, (0, 0, 255), -1)
                cv2.circle(img, index_tip, 10, (0, 0, 255), -1)
                
                # Adicionar efeito de partículas no ponto médio quando a pinça é fechada
                mid_point = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                
                if distance < 30:  # Pinça realmente fechada
                    # Adicionar muitas partículas para efeito visual
                    self.add_particles(mid_point[0], mid_point[1], (255, 0, 0), count=15)
                    
                    # Desenhar círculo pulsante
                    size = int(15 + 5 * np.sin(time.time() * 10))
                    cv2.circle(img, mid_point, size, (0, 0, 255), 3)
            
    def draw_instructions(self, img):
        """Desenha instruções na tela"""
        h, w, _ = img.shape
        
        instructions = [
            "THEREMIN VIRTUAL",
            "",
            "Mao Esquerda:",
            "- Posicao X: Controla a frequência (tom)",
            "- Posicao Y: Controla o volume",
            "",
            "Mao Direita:",
            "- Gesto de Paz (V): Muda a forma de onda",
            "- Gesto de Pinca: Toca percussao",
            "",
            "Pressione 'q' para sair"
        ]
        
        # Fundo semi-transparente
        overlay = img.copy()
        cv2.rectangle(overlay, (w-300, 10), (w-10, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        y = 30
        for line in instructions:
            cv2.putText(img, line, (w-290, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20


# --- Main Application ---
def main():
    # Configurações da Câmera e Tela
    CAM_WIDTH, CAM_HEIGHT = 1280, 720
    
    # Inicialização da Câmera
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    video.set(cv2.CAP_PROP_FPS, 30)
    
    # Inicialização do MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Inicialização dos módulos
    audio_engine = AudioEngine()
    interpreter = GestureInterpreter()
    visualizer = Visualizer(CAM_WIDTH, CAM_HEIGHT)
    
    # Iniciar o motor de áudio
    audio_engine.start()
    
    # Variáveis de estado
    frequency = 440.0
    volume = 0.0
    waveform = 'sine'
    
    print("Theremin Virtual iniciado! Pressione 'q' para sair.")
    
    try:
        # Loop principal
        while True:
            check, img = video.read()
            if not check:
                break
                
            # Processar imagem
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            h, w, _ = img.shape
            
            # Inicializar variáveis
            cursor_hand_landmarks = None
            action_hand_landmarks = None
            cursor_hand_pts = None
            action_hand_pts = None
            
            # Processar landmarks das mãos
            if results.multi_hand_landmarks:
                # Atribuir mão esquerda/direita
                if len(results.multi_hand_landmarks) == 2:
                    hand1_x = results.multi_hand_landmarks[0].landmark[0].x
                    hand2_x = results.multi_hand_landmarks[1].landmark[0].x
                    if hand1_x < hand2_x:
                        cursor_hand_landmarks = results.multi_hand_landmarks[0]
                        action_hand_landmarks = results.multi_hand_landmarks[1]
                    else:
                        cursor_hand_landmarks = results.multi_hand_landmarks[1]
                        action_hand_landmarks = results.multi_hand_landmarks[0]
                else:  # Se houver apenas uma mão, ela é a do cursor
                    cursor_hand_landmarks = results.multi_hand_landmarks[0]
                    
                # Converter landmarks para coordenadas de pixel
                if cursor_hand_landmarks:
                    cursor_hand_pts = [(int(lm.x * w), int(lm.y * h)) for lm in cursor_hand_landmarks.landmark]
                if action_hand_landmarks:
                    action_hand_pts = [(int(lm.x * w), int(lm.y * h)) for lm in action_hand_landmarks.landmark]
            
            # Processar a mão esquerda (controles contínuos)
            if cursor_hand_landmarks:
                freq_vol = interpreter.process_left_hand(cursor_hand_landmarks, w, h, audio_engine)
                if freq_vol:
                    frequency, volume = freq_vol
            else:
                # Se não há mão esquerda, diminuir o volume gradualmente
                audio_engine.set_volume(0.0)
                volume = 0.0
                
            # Processar a mão direita (gestos discretos)
            if action_hand_landmarks:
                right_hand_results = interpreter.process_right_hand(action_hand_landmarks, audio_engine)
                if right_hand_results:
                    waveform = right_hand_results['current_waveform']
            
            # Atualizar visualização
            visualizer.update_particles()
            
            # Desenhar elementos visuais
            visualizer.draw_waveform(img, waveform, frequency, volume)
            visualizer.draw_hand_tracking(img, cursor_hand_landmarks, action_hand_landmarks, mp_hands, mp_draw)
            visualizer.draw_control_points(img, cursor_hand_pts, action_hand_pts, frequency, volume, waveform)
            visualizer.draw_particles(img)
            visualizer.draw_instructions(img)
            
            # Mostrar imagem
            cv2.imshow("Theremin Virtual", img)
            
            # Verificar saída
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        # Limpeza
        audio_engine.cleanup()
        video.release()
        cv2.destroyAllWindows()
        print("Theremin Virtual finalizado!")

if __name__ == "__main__":
    main()
