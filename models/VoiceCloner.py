import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positional encodes the phoneme with sinusoidal position encoding

        Args:
            `x` (torch.Tensor)

        Returns:
            `encoded_x` (torch.Tensor): Returns the positionally encoded x
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(PhonemeEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, phonemes: torch.Tensor) -> torch.Tensor:
        """
        Transforms x to learn predictions for either pitch, durations, or energy

        Args:
            `phonemes` (torch.Tensor)

        Returns:
            `embedded_phonemes` (torch.Tensor)
        """
        return self.embedding(phonemes)  # (batch_size, seq_length, d_model)


class VariancePredictor(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super(VariancePredictor, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms x to learn predictions for either pitch, durations, or energy

        Args:
            `x` (torch.Tensor)

        Returns:
            `x` (torch.Tensor)
        """
        x = x.transpose(
            1, 2
        )  # (batch_size, seq_length, d_model)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer_norm(x.transpose(1, 2))
        x = self.dropout(x).transpose(1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.layer_norm(x.transpose(1, 2))
        x = self.dropout(x)
        x = self.fc(x).squeeze(2)
        return x


class PitchEmbedding(nn.Module):
    def __init__(self, num_bins: int, d_model: int):
        super(PitchEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_bins, d_model)

    def forward(self, quantized_pitch: torch.Tensor) -> torch.Tensor:
        """
        Embeds the quantized pitch

        Args:
            `quantized_pitch` (torch.Tensor)

        Returns:
            `embedded_quantized_pitch` (torch.Tensor)
        """
        return self.embedding(quantized_pitch)


class EnergyEmbedding(nn.Module):
    def __init__(self, num_bins: int, d_model: int):
        super(EnergyEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_bins, d_model)

    def forward(self, quantized_energy: torch.Tensor) -> torch.Tensor:
        """
        Embeds the quantized energy

        Args:
            `quantized_energy` (torch.Tensor)

        Returns:
            `embedded_quantized_energy` (torch.Tensor)
        """
        return self.embedding(quantized_energy)


class VarianceAdaptor(nn.Module):
    def __init__(
        self, d_model: int, output_dim: int, num_bins_pitch: int, num_bins_energy: int
    ):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(d_model, output_dim)
        self.pitch_predictor = VariancePredictor(d_model, output_dim)
        self.energy_predictor = VariancePredictor(d_model, output_dim)
        self.pitch_embedding = nn.Embedding(num_bins_pitch, d_model)
        self.energy_embedding = nn.Embedding(num_bins_energy, d_model)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        durations: torch.Tensor,
        use_ground_truth: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes the energy

        Args:
            `encoder_outputs` (torch.Tensor)
            `pitch` (torch.Tensor)
            `energy` (torch.Tensor)
            `durations` (torch.Tensor)
            `use_ground_truth` (boolean)
        """
        pred_durations = self.duration_predictor(encoder_outputs)

        expanded_outputs = self.expand_by_duration(
            encoder_outputs, durations if use_ground_truth else torch.exp(pred_durations) - 1
        )

        pred_pitch_spectrogram = self.pitch_predictor(expanded_outputs)
        pred_energy = self.energy_predictor(expanded_outputs)

        quantized_pitch = self.quantize_pitch(
            pred_pitch if not use_ground_truth else pitch
        )
        quantized_energy = self.quantize_energy(
            pred_energy if not use_ground_truth else energy
        )

        pitch_embeds = self.pitch_embedding(quantized_pitch)
        energy_embeds = self.energy_embedding(quantized_energy)

        adapted_outputs = expanded_outputs + pitch_embeds + energy_embeds

        return adapted_outputs, pred_durations, pred_pitch, pred_energy

    def expand_by_duration(
        self, encoder_outputs: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantizes the pitch

        Args:
            `encoder_outputs` (torch.Tensor)
            `durations` (torch.Tensor)

        Returns:
            `expanded_outputs` (torch.Tensor)
        """
        expanded = [
            torch.repeat_interleave(encoder_outputs[i], durations[i].long(), dim=0)
            for i in range(len(encoder_outputs))
        ]
        expanded_outputs = nn.utils.rnn.pad_sequence(expanded, batch_first=True)

        return expanded_outputs
    
    def quantize_pitch(
        self,
        pitch: torch.Tensor,
        num_bins: int = 256,
        min_log_f0: float = -44.34,
        max_log_f0: float = 9.71,
    ) -> torch.Tensor:
        """
        Quantizes the pitch

        Args:
            `pitch` (torch.Tensor)
            `num_bins` (int)
            `min_pitch` (int)
            `max_pitch` (int)
        
        Returns:
            `quantized` (torch.Tensor)
        """
        log_pitch = torch.log2(pitch.clam(min_pitch, max_pitch)
        bins = torch.linspace(torch.log2(min_pitch), torchlog2(max_pitch), steps=num_bins)
        quantized = torch.bucketize(log_pitch, bins)
        return quantized

    def quantize_energy(
        self,
        energy: torch.Tensor,
        num_bins: int = 256,
        min_energy: int = -1.0,
        max_energy: int = 1.0,
    ) -> torch.Tensor:
        """
        Quantizes the energy

        Args:
            `energy` (torch.Tensor)

        Returns:
            quantized (torch.Tensor)
        """
        bins = torch.linspace(min_energy, max_energy, steps=num_bins).to(energy.device)
        quantized = torch.bucketize(energy, bins)
        quantized = torch.clamp(quantized, min=0, max=num_bins - 1)
        return quantized


class PitchSpectrogramPredictor(nn.Module):
    def __init__(self, d_model: int, n_scales: int):
        super(PitchSpectrogramPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, n_scales, kernel_size=3, padding=1),
            nn.Linear(400, 5868),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms x to perform pitch spectrogram prediction

        Args:
            `x` (torch.Tensor)

        Returns:
            `x` (torch.Tensor)
        """
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.network(x)  # (batch_size, n_scales, seq_length)
        return x


class MFCCDecoder(nn.Module):
    def __init__(self, d_model: int, n_mels: int, mel_seq_len: int):
        super(MFCCDecoder, self).__init__()
        self.fc1 = nn.Linear(d_model, n_mels)
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(400, 2080)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes x into mel-spectrograms

        Args:
            `x` (torch.Tensor)

        Returns:
            `x` (torch.Tensor)
        """
        x = self.fc1(x)
        x = self.lrelu(x)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.fc2(x)  # (batch_size, n_scales, seq_length)
        return x


class VoiceCloner(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        output_dim,
        num_heads,
        num_layers,
        n_mels,
        speaker_embedding_dim,
        n_scales,
        n_pitch_bins,
        n_energy_bins,
        mel_seq_len,
    ):
        """
        Main Voice Cloner model. Goal is to learn how to synthesize voices from
        text-to-speech with few examples fine tuning.

        Args:
            `vocab_size` (int): Vocab size of the phonemes for model dictionary
            `d_model` (int): Hidden dimension size
            `output_dim` (int): Output dimension to match target size
            `num_heads` (int): Number of attention heads for the transformer encoder
            `num_layers` (int): Number of layers for the transformer encoder
            `n_mels` (int): dimensions for mel-spectrograms
            `speaker_embedding_dim` (int): Hidden dimension for the speaker embeddings
            `n_scales` (int): Number for the scale factor of the pitch spectrogram
            `n_pitch_bins` (int): Bin count for quantizing pitch
            `n_energy_bins` (int): Bin count for quantizing energy
            `mel_seq_len` (int): Maximum sequence length for the mel-spectrogram data.

        Example:
            model = VoiceCloner(vocab_size, hidden_dim, output_dim,
                    num_heads, num_layers, n_mels, speaker_embedding_dim,
                    n_scales, n_pitch_bins, n_energy_bins, mel_seq_len)
        """
        super(VoiceCloner, self).__init__()
        self.phoneme_embedding = PhonemeEmbedding(vocab_size, d_model)
        self.speaker_embedding = nn.Linear(speaker_embedding_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.variance_adaptor = VarianceAdaptor(
            d_model, output_dim, n_pitch_bins, n_energy_bins
        )
        self.pitch_spectrogram_predictor = PitchSpectrogramPredictor(d_model, n_scales)
        self.decoder = MFCCDecoder(d_model, n_mels, mel_seq_len)

    def forward(
        self,
        phonemes: torch.Tensor,
        speaker_embedding: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        durations: torch.Tensor,
        use_ground_truth: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `phonemes` (torch.Tensor): Tensor of phoneme representations of size (batch_size, max_seq_len)
            `speaker_embedding` (torch.Tensor): Tensor of phoneme representations of size (batch_size, pretrained_speaker_embedding_dim: 192)
            `pitch` (torch.Tensor): Tensor of voice pitches representations of size (batch_size, max_seq_len)
            `energy` (torch.Tensor): Tensor of voice energy representations of size (batch_size, max_seq_len)
            `durations` (torch.Tensor): Tensor of phoneme representations of size (batch_size, max_seq_len)
            `use_ground_truth` (boolean): Boolean that let's the model know to make predictions for pitch, energy, or durations

        Returns:
            `mel_outputs` (torch.Tensor): Predicted Mel-spectrogram tensors of shape (batch_size, n_mels, max_mel_seq_len)
            `pred_durations` (torch.Tensor): Predicted duration tensors of shape (batch_size, max_seq_len)
            `pred_pitch` (torch.Tensor): Predicted duration tensors of shape (batch_size, max_seq_len)
            `pred_energy` (torch.Tensor): Predicted duration tensors of shape (batch_size, max_seq_len)
            `pred_pitch_spectrogram` (torch.Tensor): Predicted voice pitch spectro gram tensors of shape (batch_size, n_scales, max_pitch_spectrogram_len)
        """
        # Embed the phoneme encoding and then use positional encoding
        # positional encoding will allow variable length text for future
        # testing
        phoneme_embeds = self.phoneme_embedding(phonemes)
        phoneme_embeds = self.positional_encoding(phoneme_embeds)

        # Embed the pre-trained speaker encodings and then use positional encoding
        # positional encoding will allow variable length speaker audio for future
        # testing
        speaker_embedded = self.speaker_embedding(speaker_embedding).unsqueeze(1)

        # Phoneme embeddings are conditioned on the speaker embeddings to help the model synthesize
        # voice cloning
        conditioned_inputs = phoneme_embeds + speaker_embedded

        # Encodes the conditioned phoneme embeddings and speaker embeddings with the transformer encoder
        encoder_outputs = self.encoder(conditioned_inputs)

        adapted_outputs, pred_durations, pred_pitch, pred_energy = (
            self.variance_adaptor(
                encoder_outputs, pitch, energy, durations, use_ground_truth
            )
        )

        pred_pitch_spectrogram = self.pitch_spectrogram_predictor(adapted_outputs)
        mel_outputs = self.decoder(adapted_outputs)

        return (
            mel_outputs,
            pred_durations,
            pred_pitch,
            pred_energy,
            pred_pitch_spectrogram,
        )
