# M2QA Benchmark Dataset
The M2QA benchmark dataset consists of 13,500 SQuAD 2.0-style question-answer instances, divided evenly across nine language-domain combination pairs (1500 instances each). 40% of the data are unanswerable questions, 60% are answerable. We provide 7500 additional training examples.

Following [Jacovi et al. (2023)](https://aclanthology.org/2023.emnlp-main.308/), we encrypt the validation data to prevent leakage of the dataset into LLM training datasets. Additional training examples [training data](Additional_Training_data) come from the same datasets (train split instead of test split). Also uploaded on Hugging Face. And since it's training data, it is unencrypted.

To unencrypt the data, execute:

```bash
unzip -P m2qa german.zip
unzip -P m2qa chinese.zip
unzip -P m2qa turkish.zip
```

You can then easily load it, e.g. like this:

```python
from datasets import load_dataset

LANGUAGES = ["german", "chinese", "turkish"]
DOMAINS = ["news", "creative_writing", "product_reviews"]

def load_m2qa_dataset(args: argparse.Namespace):
    m2qa_dataset = {}
    for language in LANGUAGES:
        m2qa_dataset[language] = load_dataset(
            "json",
            data_files={domain: f"m2qa_dataset/{language}/{domain}.json" for domain in DOMAINS},
        )

    return m2qa_dataset
```

### Via Hugging Face
The dataset is also available via Hugging Face datasets: [https://huggingface.co/datasets/UKPLab/m2qa](https://huggingface.co/datasets/UKPLab/m2qa)
Follow the instructions there to see how easily you can load the data & evaluate models with it.

## Licences
The contextes stem from sources with open licenses:

| Language | Domain           | Multiple Passages | Datasource                                                                                | License                                                                              |
| -------- | ---------------- | ----------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| German   | product reviews  | no                | Amazon Reviews  ([Keung et al., 2020](https://aclanthology.org/2020.emnlp-main.369/))     | Usage permitted by Amazon for academic research \[1\].                               |
|          | news             | yes               | 10kGNAD \[2\]                                                                             | CC BY-NC-SA 4.0                                                                      |
|          | creative writing | yes               | Gutenberg Corpus ([Gerlach and Font-Clos, 2018](https://www.mdpi.com/1099-4300/22/1/126)) | Manually selected text passages from open-license books.                             |
| Turkish  | product reviews  | no                | Turkish product reviews \[3\]                                                             | CC BY-SA 4.0                                                                         |
|          | news             | yes               | BilCat ([Toraman et al., 2011](https://ieeexplore.ieee.org/document/5946096))             | MIT License                                                                          |
|          | creative writing | yes               | Wattpad \[4\]                                                                             | Manually selected text passages from Creative Commons or Public Domain publications. |
| Chinese  | product reviews  | no                | Amazon Reviews ([Keung et al., 2020](https://aclanthology.org/2020.emnlp-main.369/))      | Usage permitted by Amazon for academic research [^1].                                |
|          | news             | yes               | CNewSum (Wang et al., 2021)                                                               | MIT License                                                                          |
|          | creative writing | yes               | Wattpad \[4\]                                                                             | Manually selected text passages from Creative Commons or Public Domain publications. |

- \[1\]: [https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt)
- \[2\]: https://github.com/tblock/10kGNAD using the One Million Posts dataset by [Schabus et al. (2017)](https://example.com/schabus2017one)  
- \[3\]: https://huggingface.co/datasets/turkish_product_reviews  
- \[4\]: https://www.wattpad.com/  

## License
The M2QA dataset is distributed under the [CC-BY-ND 4.0 license](LICENSE). For further information, refer to: [https://creativecommons.org/licenses/by-nd/4.0/legalcode](https://creativecommons.org/licenses/by-nd/4.0/legalcode)

Following [Jacovi et al. (2023)](https://aclanthology.org/2023.emnlp-main.308/), we decided to publish with a "No Derivatives" license to mitigate the risk of data contamination of crawled training datasets.
