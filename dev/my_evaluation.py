"""
This code is based on the TorchKGE library
(https://github.com/torchkge-team/torchkge), originally developed by
Armand Boschin <aboschin@enst.fr>, and has been modified
by Galadriel Brière.
"""

from torch import empty, zeros, cat
from tqdm.autonotebook import tqdm
from torchkge.utils import DataLoader, get_rank, filter_scores
from torchkge.exceptions import NotYetEvaluatedError
from torchkge.utils.modeling import get_true_targets

def filter_scores_with_candidates(scores, dictionary, key1, key2, true_idx, candidates):
    filt_scores = scores.clone()
    b_size = scores.shape[0]

    # Set everything to -inf
    filt_scores[:] = -float("inf")

    # Set scores only for candidates
    filt_scores[:, candidates] = scores[:, candidates]

    # Remove known true targets (as d'habitude)
    for i in range(b_size):
        true_targets = get_true_targets(dictionary, key1, key2, true_idx, i)
        if true_targets is not None:
            filt_scores[i][true_targets] = -float("inf")

    return filt_scores


class LinkPredictionEvaluator(object):
    """Evaluate performance of given embedding using link prediction method.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    knowledge_graph: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true head when all entities
        are ranked as possible replacement of the head entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true tail when all entities
        are ranked as possible replacement of the tail entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filt_rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.

    """

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()

        self.evaluated = False

    def evaluate(self, b_size, candidate_idx=None, mode="tail", verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.
        candidate_idx: torch.LongTensor or None
            List of candidate entity indices (only used when restrict to subset).
        mode: str
            Mode of evaluation, either "tail" or "head". Considered only when candidate_idx is not None.
        """
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size, use_cuda='batch')
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()
        else:
            dataloader = DataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Link prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            h_emb, t_emb, r_emb, candidates = self.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=True)

            if candidate_idx is None:
                scores = self.model.inference_scoring_function(h_emb, candidates, r_emb)
                filt_scores = filter_scores(scores, self.kg.dict_of_tails, h_idx, r_idx, t_idx)
                self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(scores, t_idx).detach()
                self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()
                self.tail_evaluated = True


                scores = self.model.inference_scoring_function(candidates, t_emb, r_emb)
                filt_scores = filter_scores(scores, self.kg.dict_of_heads, t_idx, r_idx, h_idx)
                self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(scores, h_idx).detach()
                self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()
                self.head_evaluated = True

            else:
                if mode == "tail":
                    scores = self.model.inference_scoring_function(h_emb, candidates, r_emb)
                    filt_scores = filter_scores_with_candidates(scores, self.kg.dict_of_tails, h_idx, r_idx, t_idx, candidate_idx)
                    self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(scores, t_idx).detach()
                    self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()
                    self.tail_evaluated = True
                    self.head_evaluated = False
                elif mode == 'head':
                    scores = self.model.inference_scoring_function(candidates, t_emb, r_emb)
                    filt_scores = filter_scores_with_candidates(
                        scores, self.kg.dict_of_heads, t_idx, r_idx, h_idx, candidate_idx
                    )
                    self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(scores, h_idx).detach()
                    self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()
                    self.head_evaluated = True
                    self.tail_evaluated = False
                else:
                    raise ValueError("When using candidate_idx, predict must be either 'head' or 'tail'")
                    
        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        
        ranks, filt_ranks = [], []

        if self.head_evaluated and not self.tail_evaluated:
            ranks.append(self.rank_true_heads.float().mean().item())
            filt_ranks.append(self.filt_rank_true_heads.float().mean().item())
        elif self.tail_evaluated and not self.head_evaluated:
            ranks.append(self.rank_true_tails.float().mean().item())
            filt_ranks.append(self.filt_rank_true_tails.float().mean().item())
        else:
            ranks.append(self.rank_true_heads.float().mean().item())
            ranks.append(self.rank_true_tails.float().mean().item())
            filt_ranks.append(self.filt_rank_true_heads.float().mean().item())
            filt_ranks.append(self.filt_rank_true_tails.float().mean().item())

        return sum(ranks) / len(ranks), sum(filt_ranks) / len(filt_ranks)

    def hit_at_k_heads(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        if not self.head_evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated on heads ')
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

        return head_hit.item(), filt_head_hit.item()

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        if not self.tail_evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated on tails ')
        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

        return tail_hit.item(), filt_tail_hit.item()

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        
        hits, filt_hits = [], []

        if self.head_evaluated and not self.tail_evaluated:
            hits.append((self.rank_true_heads <= k).float().mean())
            filt_hits.append((self.filt_rank_true_heads <= k).float().mean().item())

        elif self.tail_evaluated and not self.head_evaluated:
            hits.append((self.rank_true_tails <= k).float().mean())
            filt_hits.append((self.filt_rank_true_tails <= k).float().mean().item())

        else:
            hits.append((self.rank_true_heads <= k).float().mean())
            hits.append((self.rank_true_tails <= k).float().mean())
            filt_hits.append((self.filt_rank_true_heads <= k).float().mean().item())
            filt_hits.append((self.filt_rank_true_tails <= k).float().mean().item())

        return sum(hits) / len(hits), sum(filt_hits) / len(filt_hits)

    def mrr(self):
        """
  
        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        if self.head_evaluated and not self.tail_evaluated:
            head_mrr = (self.rank_true_heads.float() ** -1).mean()
            filt_head_mrr = (self.filt_rank_true_heads.float() ** -1).mean()
            return head_mrr.item(), filt_head_mrr.item()

        elif self.tail_evaluated and not self.head_evaluated:
            tail_mrr = (self.rank_true_tails.float() ** -1).mean()
            filt_tail_mrr = (self.filt_rank_true_tails.float() ** -1).mean()
            return tail_mrr.item(), filt_tail_mrr.item()

        else:
            head_mrr = (self.rank_true_heads.float()**(-1)).mean()
            tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
            filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
            filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()

            return ((head_mrr + tail_mrr).item() / 2,
                    (filt_head_mrr + filt_tail_mrr).item() / 2)

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))
