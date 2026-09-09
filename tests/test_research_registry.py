import copy
import unittest

from scripts.research_registry import claim, normalize_identifier, validate


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.data = {'papers': [{'paper_id': 'a', 'identifiers': ['arxiv:2602.20945', 'doi:10.1234/example'], 'qc_status': 'not_sampled'}, {'paper_id': 'b', 'identifiers': ['arxiv:2403.12968'], 'qc_status': 'not_sampled'}]}

    def test_arxiv_versions_and_formats_share_work(self):
        for value in ['2602.20945v3', 'https://arxiv.org/pdf/2602.20945v1.pdf', 'https://arxiv.org/html/2602.20945v2?utm_source=test']:
            self.assertEqual(normalize_identifier(value), 'arxiv:2602.20945')

    def test_doi_normalization(self):
        self.assertEqual(normalize_identifier('https://doi.org/10.18653/V1/2024.findings-acl.57?ref=x'), 'doi:10.18653/v1/2024.findings-acl.57')

    def test_alias_collision_rejected(self):
        self.data['papers'][1]['identifiers'].append('https://arxiv.org/abs/2602.20945v2')
        self.assertTrue(any('identifier collision' in x for x in validate(self.data)))

    def test_second_active_read_rejected_without_mutation(self):
        claim(self.data, 'a', 'first', 'luna1', 'methods', 'v1')
        before = copy.deepcopy(self.data)
        with self.assertRaises(ValueError):
            claim(self.data, 'a', 'second', 'luna2', 'experiments', 'v1')
        self.assertEqual(before, self.data)

    def test_task_id_cannot_be_reused_for_another_paper(self):
        claim(self.data, 'a', 'task1', 'luna1', 'main_body', 'v1')
        with self.assertRaises(ValueError):
            claim(self.data, 'b', 'task1', 'luna2', 'main_body', 'v1')

    def test_similar_titles_do_not_merge_different_identifiers(self):
        for paper in self.data['papers']:
            paper['title'] = 'Efficient Reasoning'
        self.assertEqual(validate(self.data), [])
        self.assertEqual(len(self.data['papers']), 2)

    def test_unsampled_is_valid_but_claimed_sample_needs_evidence(self):
        self.assertEqual(validate(self.data), [])
        self.data['papers'][0]['qc_status'] = 'sample_pass'
        self.assertTrue(validate(self.data))
        self.data['papers'][0]['qc_record'] = 'meta/research-map-v3/qc-samples.json#sample-1'
        self.assertEqual(validate(self.data), [])


if __name__ == '__main__':
    unittest.main()
