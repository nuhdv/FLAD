import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List
import math
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
from .algorithm_utils import Algorithm, PyTorchUtils, get_sub_seqs, get_train_data_loaders
from sklearn.decomposition import PCA
from tqdm import trange
import logging
from .cross_stitch import CrossStitchNetwork

class FLAD(Algorithm, PyTorchUtils):
    def __init__(self, name: str='FLAD', num_epochs: int=10, batch_size: int=32, lr: float=1e-3, sequence_length:
                 int=55, num_channels: List=None, kernel_size: int=5, dropout: float=0.2,
                 train_val_percentage: float=0.10, seed: int=None, gpu: int=None, details=False, patience: int=2,
                 stride: int=1, out_dir=None, pca_comp=None):

        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        np.random.seed(seed)
        self.torch_save = True
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.patience = patience
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.pca_comp = pca_comp
        self.additional_params = dict()

    def get_model(self, inputs):
        backbone_dict, decoder_dict = {}, {}
        cross_stitch_kwargs = {'alpha': 0.8, 'beta': 0.2, 'stages': ['layer1', 'layer2', 'layer3'],
                               'channels': {'layer1': 64, 'layer2': 128, 'layer3': 256},
                               'num_channels': self.num_channels}

        TCN = SingleTaskModel(TCN_encoder(num_inputs=inputs, num_channels=self.num_channels, kernel_size=self.kernel_size, dropout=0.2),
                              TCN_decoder(num_inputs=inputs, num_channels=list(reversed(self.num_channels)), kernel_size=self.kernel_size, dropout=0.2), 'reconstruct')
        TCN = torch.nn.DataParallel(TCN)
        backbone_dict['reconstruct'] = TCN.module.encoder
        decoder_dict['reconstruct'] = TCN.module.decoder

        Trans = SingleTaskModel(Trans_encoder(num_inputs=inputs, feature_size=self.num_channels[-1], num_channels=self.num_channels, num_layers=1, dropout=0.1),
                                Trans_decoder(num_inputs=inputs, feature_size=self.num_channels[-1], num_layers=1, dropout=0.1), 'predict')
        Trans = torch.nn.DataParallel(Trans)
        backbone_dict['predict'] = Trans.module.encoder
        decoder_dict['predict'] = Trans.module.decoder
        model = CrossStitchNetwork(['reconstruct', 'predict'], torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), **cross_stitch_kwargs)
        return model

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            # Project input data on a limited number of principal components
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
            pca.fit(data)
            self.additional_params["pca"] = pca
            data = pca.transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride)
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.model = self.get_model(inputs=data.shape[1])
        self.model, train_loss, val_loss, val_reconstr_errors, val_predict_errors, best_val_loss = \
            fit_with_early_stopping(train_loader, train_val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, ret_best_val_loss=True)
        self.additional_params["train_loss_per_epoch"] = train_loss
        self.additional_params["val_loss_per_epoch"] = val_loss
        self.additional_params['val_reconstr_errors'] = val_reconstr_errors
        self.additional_params['val_predict_errors'] = val_predict_errors
        self.additional_params["best_val_loss"] = best_val_loss

    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            data = self.additional_params["pca"].transform(data)
        sequences = get_sub_seqs(data, seq_len=self.sequence_length, stride=1)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model, test_loader, return_output=True)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic

class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}

def fit_with_early_stopping(train_loader, val_loader, pytorch_module, patience, num_epochs, lr, last_t_only=True, ret_best_val_loss=False):
    """
    :param train_loader: the pytorch data loader for the training set
    :param val_loader: the pytorch data loader for the validation set
    :param pytorch_module:
    :param patience:
    :param num_epochs: the maximum number of epochs for the training
    :param lr: the learning rate parameter used for optimization
    :return: trained module, avg train and val loss per epoch, final loss on train + val data per channel
    """
    pytorch_module = torch.nn.DataParallel(pytorch_module)
    pytorch_module = pytorch_module.cuda()
    optimizer = torch.optim.Adam(pytorch_module.parameters(), lr=lr)
    epoch_wo_improv = 0
    pytorch_module.train()
    train_loss_by_epoch = []
    val_loss_by_epoch = []
    best_val_loss = None
    best_params = pytorch_module.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            pytorch_module.train()
            train_loss = []
            for n, ts_batch in enumerate(train_loader):
                ts_batch = ts_batch.float().cuda()
                target = ts_batch[:, -1, :].view(ts_batch.shape[0], 1, ts_batch.shape[2])
                target = target.float().cuda()
                output = pytorch_module(ts_batch, target)
                reconstruct = output['reconstruct']
                predict = output['predict']
                if last_t_only:
                    loss1 = nn.MSELoss(reduction="mean")(reconstruct, target)
                    loss2 = nn.MSELoss(reduction="mean")(predict, target)
                    loss = 0.5 * loss1 + 0.5 * loss2
                else:
                    loss1 = nn.MSELoss(reduction="mean")(reconstruct, ts_batch)
                    loss2 = nn.MSELoss(reduction="mean")(predict, ts_batch)
                    loss = 0.5 * loss1 + 0.5 * loss2
                pytorch_module.zero_grad()
                loss.backward()
                optimizer.step()
                # multiplying by length of batch to correct accounting for incomplete batches
                train_loss.append(loss.item() * len(target))

            train_loss = np.mean(train_loss)/train_loader.batch_size
            train_loss_by_epoch.append(train_loss)

            # Get Validation loss
            pytorch_module.eval()
            val_loss = []
            with torch.no_grad():
                for n, ts_batch in enumerate(val_loader):
                    ts_batch = ts_batch.float().cuda()
                    target = ts_batch[:, -1, :].view(ts_batch.shape[0], 1, ts_batch.shape[2])
                    target = target.float().cuda()
                    output = pytorch_module(ts_batch, target)
                    reconstruct = output['reconstruct']
                    predict = output['predict']
                    if last_t_only:
                        loss1 = nn.MSELoss(reduction="mean")(reconstruct, target)
                        loss2 = nn.MSELoss(reduction="mean")(predict, target)
                        loss = 0.5 * loss1 + 0.5 * loss2
                    else:
                        loss1 = nn.MSELoss(reduction="mean")(reconstruct, ts_batch)
                        loss2 = nn.MSELoss(reduction="mean")(predict, ts_batch)
                        loss = 0.5 * loss1 + 0.5 * loss2
                    val_loss.append(loss.item()*len(target))
            val_loss = np.mean(val_loss)/val_loader.batch_size
            val_loss_by_epoch.append(val_loss)
            print(f'{epoch}/{num_epochs}', val_loss)
            best_val_loss_epoch = np.argmin(val_loss_by_epoch)
            if best_val_loss_epoch == epoch:
                # any time a new best is encountered, the best_params will get replaced
                best_params = pytorch_module.state_dict()
                best_val_loss = val_loss
            # Check for early stopping by counting the number of epochs since val loss improved
            if epoch > 0 and val_loss >= val_loss_by_epoch[-2]:
                epoch_wo_improv += 1
            else:
                epoch_wo_improv = 0
        else:
            # early stopping is applied
            pytorch_module.load_state_dict(best_params)
            break
    pytorch_module.eval()
    val_reconstr_errors = []
    val_predict_errors = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().cuda()
            target = ts_batch[:, -1, :].view(ts_batch.shape[0], 1, ts_batch.shape[2])
            target = target.float().cuda()
            output = pytorch_module(ts_batch, target)
            reconstruct = output['reconstruct']
            predict = output['predict']
            error1 = nn.L1Loss(reduction="none")(reconstruct, target)
            error2 = nn.L1Loss(reduction="none")(predict, target)
            val_reconstr_errors.append(error1.cpu().numpy())
            val_predict_errors.append(error2.cpu().numpy())
    if len(val_reconstr_errors) > 0:
        val_reconstr_errors = np.concatenate(val_reconstr_errors)
    if ret_best_val_loss:
        return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors, val_predict_errors, best_val_loss
    return pytorch_module, train_loss_by_epoch, val_loss_by_epoch, val_reconstr_errors, val_predict_errors
def predict_test_scores(pytorch_module, test_loader, return_output=False):
    pytorch_module.eval()
    reconstr_scores = []
    outputs_array = []
    with torch.no_grad():
        for ts_batch in test_loader:
            ts_batch = ts_batch.float().cuda()
            target = ts_batch[:, -1, :].view(ts_batch.shape[0], 1, ts_batch.shape[2])
            target = target.float().cuda()
            output = pytorch_module(ts_batch, target)
            reconstruct = output['reconstruct']
            predict = output['predict']
            error1 = nn.L1Loss(reduction='none')(reconstruct, target)
            error2 = nn.L1Loss(reduction='none')(predict, target)

            error = 0.5 * error1 + 0.5 * error2
            reconstr_scores.append(error.cpu().numpy())
            outputs_array.append(predict.cpu().numpy())

    reconstr_scores = np.concatenate(reconstr_scores)
    outputs_array = np.concatenate(outputs_array)
    reconstr_scores = reconstr_scores[:, -1]
    outputs_array = outputs_array[:, -1]
    multivar = (len(reconstr_scores.shape) > 1)
    if multivar:
        padding = np.zeros((len(ts_batch[0]) - 1, reconstr_scores.shape[-1]))
    else:
        padding = np.zeros(len(ts_batch[0]) - 1)
    reconstr_scores = np.concatenate([padding, reconstr_scores])
    outputs_array = np.concatenate([padding, outputs_array])
    print('padding', reconstr_scores.shape, padding.shape)
    if return_output:
        return_vars = (reconstr_scores, outputs_array)
    else:
        return_vars = reconstr_scores
    return return_vars

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class pad1d(nn.Module):
    def __init__(self, pad_size):
        super(pad1d, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return torch.cat([x, x[:, :, -self.pad_size:]], dim = 2).contiguous()

class TemporalBlockTranspose(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
        dropout=0.2):
        super(TemporalBlockTranspose, self).__init__()
        self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.pad1 = pad1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.pad2 = pad1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.dropout1, self.relu1, self.pad1, self.conv1,
            self.dropout2, self.relu2, self.pad2, self.conv2)
        self.downsample = nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.src_mask = None
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        return self.dropout(x + self.pe[:x.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x.permute(2, 0, 1)

class TCN_encoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_encoder, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*self.layers)

    def forward_stage(self, x, stage):
        assert (stage in ['layer1', 'layer2', 'layer3', 'layer4'])
        if stage == 'layer1':
            x = self.layers[0](x)
            return x
        elif stage == 'layer2':
            x = self.layers[1](x)
            return x
        elif stage == 'layer3':
            x = self.layers[2](x)
            return x

    def forward(self, x):
        out = x.permute(0, 2, 1)
        return self.network(out)

class TCN_decoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_decoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # no dilation in decoder
            in_channels = num_channels[i]
            out_channels = num_inputs if i == (num_levels - 1) else num_channels[i + 1]
            dilation_size = 2 ** (num_levels - 1 - i)
            padding_size = (kernel_size - 1) * dilation_size
            layers += [TemporalBlockTranspose(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=padding_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.fcn = nn.Sequential(nn.Linear(num_channels[0], num_inputs), nn.Sigmoid())

    def forward(self, x, tgt):
        out = self.network(x)
        out = out.permute(0, 2, 1)
        return out[:, -1].view(out.shape[0], 1, out.shape[2])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weight


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Trans_encoder(nn.Module):
    def __init__(self, num_inputs, feature_size=512, num_channels=[64, 128, 256], num_layers=1, dropout=0.1):
        super(Trans_encoder, self).__init__()

        self.src_mask = None
        self.embedding = TokenEmbedding(c_in=num_inputs * 2, d_model=feature_size)
        # self.embed = TokenEmbedding(c_in=num_inputs, d_model=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, dropout=0.1)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layer1 = self._make_layer(inputs=num_inputs, feature_size=num_channels[0], num_layers=num_layers,
                                       dropout=dropout)
        self.layer2 = self._make_layer(inputs=num_channels[0], feature_size=num_channels[1], num_layers=num_layers,
                                       dropout=dropout)
        self.layer3 = self._make_layer(inputs=num_channels[1], feature_size=num_channels[2], num_layers=num_layers,
                                       dropout=dropout)

    def _make_layer(self, inputs, feature_size, num_layers, dropout):
        # layers = []
        embedding = TokenEmbedding(c_in=inputs, d_model=feature_size)
        pos_encoder = PositionalEncoding(feature_size, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return nn.Sequential(embedding, pos_encoder, transformer_encoder)

    def forward_stage(self, x, stage):
        assert(stage in ['layer1', 'layer2', 'layer3'])
        layer = getattr(self, stage)
        x ,w = layer(x)
        return x.permute(1, 2, 0), w

    def forward(self, src, c):
        src = self.embedding(torch.cat((src, c), dim=2))
        src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output.permute(1, 2, 0)

class Trans_decoder(nn.Module):
    def __init__(self, num_inputs, feature_size=512, num_layers=1, dropout=0.1):
        super(Trans_decoder, self).__init__()

        self.embed = TokenEmbedding(c_in=num_inputs, d_model=feature_size)
        decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_inputs)
        self.fcn = nn.Sequential(nn.Linear(feature_size, num_inputs), nn.Sigmoid())

    def forward(self, output, tgt):
        tgt = tgt.permute(0, 2, 1)
        out = self.transformer_decoder(self.embed(tgt), output.permute(2, 0, 1))
        out = self.decoder(out)
        return out.permute(1, 0, 2)[:, -1].view(out.shape[1], 1, out.shape[2])