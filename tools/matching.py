# --------------------------------------------------------
# SIGMA++: Improved Semantic-complete Graph Matching for Domain Adaptive Object Detection
# Written by Wuyang Li
# Based on https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/condgraph.py
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
# from .loss import make_prototype_evaluator
# from fcos_core.layers import BCEFocalLoss, MultiHeadAttention, HyperGraph
import sklearn.cluster as cluster
# from fcos_core.modeling.discriminator.layer import GradientReversal
# import logging
import random
import logging

# seed = 0
# torch.manual_seed(seed)
# random.seed(seed)
# # np.random.seed(args.seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logger = logging.getLogger("fcos_core.trainer")

class GModule0(torch.nn.Module):

    def __init__(self, num_classes):
        super(GModule0, self).__init__()
        self.num_classes = num_classes

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=2048)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        # if cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'L1':
        #     self.matching_loss = nn.L1Loss(reduction='sum')
        # elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'MSE':
        #     self.matching_loss = nn.MSELoss(reduction='sum')
        # elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'BCE':
        
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')
        self.matching_loss = nn.MSELoss(reduction='mean')
        # self.matching_loss = nn.MSELoss(reduction='sum')
        # self.matching_loss = BCEFocalLoss(reduction='elementwise_mean')
        # self.matching_loss = nn.L1Loss(reduction='sum')

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d=256):
        super(Affinity, self).__init__()
        
        self.d = d

        # self.fc_M = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1))
        
        self.fc_M = nn.Sequential(
            nn.Linear(1024, 1),
            # nn.ReLU(),
            # nn.Linear(512, 1)
            )
        
        self.fc_M0 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

        self.project_sr = nn.Linear(2048, 512,bias=False)
        self.project_tg = nn.Linear(2048, 512,bias=False)

        # self.project_sr = nn.Linear(2048, 1024,bias=False)
        # self.project_tg = nn.Linear(2048, 1024,bias=False)
        # self.reset_parameters()


    def reset_parameters(self):

        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)


        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)

        # The common GM design doesn;t work!!
        # stdv = 1. / math.sqrt(self.d)
        # self.A.data.uniform_(-stdv, stdv)
        # self.A.data += torch.eye(self.d).cuda()
        # nn.init.normal_(self.project_2.weight, std=0.01)
        # nn.init.normal_(self.project2.weight, std=0.01)
        # nn.init.constant_(i.bias, 0)
    def forward(self, X, Y):

        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2, C = Y.size()

        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)

        # M = torch.cat([X_k, Y_k], dim=-1)
        # M = self.fc_M(M).squeeze()


        M = X_k - Y_k
        M = self.fc_M0(M).squeeze()

        # The common GM design doesn;t work!!

        # M = self.affinity_pred(M[None,]).squeeze()
        # M_r = self.fc_M(M_r).squeeze()
        # M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous()) / 2)
        # M = torch.matmul(M, Y.transpose(0, 1).contiguous())
        return M

class GModule(torch.nn.Module):

    def __init__(self, num_classes):
        super(GModule, self).__init__()
        self.num_classes = num_classes

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=2048)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        # if cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'L1':
        #     self.matching_loss = nn.L1Loss(reduction='sum')
        # elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'MSE':
        #     self.matching_loss = nn.MSELoss(reduction='sum')
        # elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'BCE':
        
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')
        self.matching_loss = nn.MSELoss(reduction='mean')
        # self.matching_loss = nn.MSELoss(reduction='sum')
        # self.matching_loss = BCEFocalLoss(reduction='elementwise_mean')
        # self.matching_loss = nn.L1Loss(reduction='sum')
        

    def forward(self, nodes_1, nodes_2, labels_1, labels_2, matching_cfg='o2o', with_quadratic_matching=False):
        loss = 0
        if matching_cfg != 'none':
            matching_loss_affinity, affinity = self._forward_aff(nodes_1, nodes_2, labels_1, labels_2, matching_cfg=matching_cfg)
            loss = loss + matching_loss_affinity

            # if with_quadratic_matching:
            #     matching_loss_quadratic = self._forward_qu(nodes_1, nodes_2, edges_1.detach(), edges_2.detach(), affinity)
            #     loss = loss + matching_loss_quadratic
        return loss

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        k = 20  # conduct clustering when we have enough graph nodes
        for cls in sr_labels.unique().long():
            bs = sr_nodes[sr_labels == cls].detach()

            if len(bs) > k and self.with_cluster_update:
                # TODO Use Pytorch-based GPU version
                sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                seed_cls = self.sr_seed[cls]
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                indx = (indx == indx[0])[1:]
                bs = bs[indx].mean(0)
            else:
                bs = bs.mean(0)

            momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
            self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

        if tg_nodes is not None:
            for cls in tg_labels.unique().long():
                bs = tg_nodes[tg_labels == cls].detach()
                if len(bs) > k and self.with_cluster_update:
                    seed_cls = self.tg_seed[cls]
                    sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                    assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                    indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                    indx = (indx == indx[0])[1:]
                    bs = bs[indx].mean(0)
                else:
                    bs = bs.mean(0)
                momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
                self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2, matching_cfg='o2o'):
        # print('------matching ')
        if matching_cfg == 'o2o':
            

            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_iter(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float()
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TODO Find a better reduction strategy
            # print(TP_samples)
            # print(TP_samples.sum(1))
            # print(TP_target.sigmoid())
            # print(TP_target.sigmoid().sum(1))
            TP_loss = self.matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
            matching_loss = TP_loss + FP_loss

        elif matching_cfg == 'm2m':  # Refer to the Appendix
            print(nodes_1.shape, nodes_2.shape)
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            logger.info(matching_target)
            logger.info(M.sigmoid())
            logger.info(str(matching_target.shape) +  str(M.shape))

            M = M.sigmoid().reshape(-1)
            matching_target = matching_target.reshape(-1)

            TP_index = matching_target == 1
            FP_index = matching_target == 0

            TP_samples, TP_target = M[TP_index], matching_target[TP_index]
            FP_samples, FP_target = M[FP_index], matching_target[FP_index]

            # down sample
            TP_len = len(TP_target)
            TP_FP_ratio = 1.
            max_FP_len = min(len(FP_target), int(TP_FP_ratio * TP_len))
            FP_index = list(range(len(FP_target)))
            random.shuffle(FP_index)
            FP_index = FP_index[:max_FP_len]
            FP_samples, FP_target = FP_samples[FP_index], FP_target[FP_index]

            TP_loss = self.matching_loss(TP_samples, TP_target)
            FP_loss = self.matching_loss(FP_samples, FP_target)
            matching_loss = TP_loss + FP_loss


            # indx = (M * TP_mask).max(-1)[1]
            # TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            # TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            # FP_samples = M[matching_target == 0].view(-1, 1)
            # FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # # TODO Find a better reduction strategy
            
            
            # TP_loss = self.matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
            # FP_loss = self.matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
            # matching_loss = TP_loss + FP_loss


            # matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_qu(self, nodes_1, nodes_2, edges_1, edges_2, affinity):

        if self.with_hyper_graph:

            # hypergraph matching (high order)
            translated_indx = list(range(1, self.num_hyper_edge))+[int(0)]
            mathched_index = affinity.argmax(0)
            matched_node_1 = nodes_1[mathched_index]
            matched_edge_1 = edges_1.t()[mathched_index]
            matched_edge_1[matched_edge_1 > 0] = 1

            matched_node_2 =nodes_2
            matched_edge_2 =edges_2.t()
            matched_edge_2[matched_edge_2 > 0] = 1
            n_nodes = matched_node_1.size(0)

            angle_dis_list = []
            for i in range(n_nodes):
                triangle_1 = nodes_1[matched_edge_1[i, :].bool()]  # 3 x 256
                triangle_1_tmp = triangle_1[translated_indx]
                # print(triangle_1.size(), triangle_1_tmp.size())
                sin1 = torch.sqrt(1.- F.cosine_similarity(triangle_1, triangle_1_tmp).pow(2)).sort()[0]
                triangle_2 = nodes_2[matched_edge_2[i, :].bool()]  # 3 x 256
                triangle_2_tmp = triangle_2[translated_indx]
                sin2 = torch.sqrt(1.- F.cosine_similarity(triangle_2, triangle_2_tmp).pow(2)).sort()[0]
                angle_dis = (-1 / self.angle_eps  * (sin1 - sin2).abs().sum()).exp()
                angle_dis_list.append(angle_dis.view(1,-1))

            angle_dis_list = torch.cat(angle_dis_list)
            loss = angle_dis_list.mean()
        else:
            # common graph matching (2nd order)
            R = torch.mm(edges_1, affinity) - torch.mm(affinity, edges_2)
            loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

    def sinkhorn_iter(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()
        return log_alpha

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long(), :].cuda()

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.clamp(torch.sigmoid(_input), min=0.00001)
        # pt = torch.clamp(pt, max=0.99999)
        pt = _input
        alpha = self.alpha

        # pos = torch.nonzero(target[:,1] > 0).squeeze(1).numel()
        # print(pos)

        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'pos':
            loss = torch.sum(loss) / (2 * pos)

        return loss
    
def reparametrize(mean,var): # 采样器方法：对方差(lg_var)进行还原，并从高斯分布中采样，将采样数值映射到编码器输出的数据分布中。
    # std = var.exp().sqrt()
    std = var.sqrt()
    # print('------')
    # print(var)
    # print(std)
    # print(std_0)
    # torch.FloatTensor(std.size())的作用是，生成一个与std形状一样的张量。然后，调用该张量的normal_()方法，系统会对该张量中的每个元素在标准高斯空间（均值为0、方差为1）中进行采样。
    eps = torch.FloatTensor(std.size()).normal_().cuda() # 随机张量方法normal_()，完成高斯空间的采样过程。
    return eps.mul(std).add_(mean)