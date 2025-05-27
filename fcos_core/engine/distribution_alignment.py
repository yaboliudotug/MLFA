

def category_level_alignment(feat_ext2, pseudo_label2):
    for label in pseudo_label2.unique():                        
        feat_ext_per_category = feat_ext2[pseudo_label2 == label, :]

        # 避免协方差矩阵计算错误，好像和矩阵维度有关，样本数量不能大于128？
        if feat_ext_per_category.shape[0] > 100:
            max_length = 100
            feat_ext_per_category = feat_ext_per_category[:max_length]

        b = feat_ext_per_category.shape[0]
        ema_n[label] += b
        alpha = 1. / ema_length if ema_n[label] > ema_length else 1. / ema_n[label]

        ema_ext_mu_that = ema_ext_mu[label, :]
        ema_ext_cov_that = ema_ext_cov[label, :, :]
        delta_pre = feat_ext_per_category - ema_ext_mu_that

        delta = alpha * delta_pre.sum(dim=0)
        tmp_mu = ema_ext_mu_that + delta
        tmp_cov = ema_ext_cov_that + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_cov_that) - delta[:, None] @ delta[None, :]

        with torch.no_grad():
            ema_ext_mu[label, :] = tmp_mu.detach()
            ema_ext_cov[label, :, :] = tmp_cov.detach()
        
        if ema_n[label] >= 16:
            try:
                source_domain = torch.distributions.MultivariateNormal(ext_src_mu[label, :], ext_src_cov[label, :, :] + template_ext_cov)
                target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
                loss_kl_category += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) / 2.0
            except:
                logger.info('wrong kl category label: ' + str(label))
                continue