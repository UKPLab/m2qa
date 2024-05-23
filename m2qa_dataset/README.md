# M2QA Benchmark Dataset
**⚠️⚠️⚠️ This is the only yet unfinished section. TODO: write this section & encrypt the data.**
- Why did we encrypt the data?
- How to load it?
  - Before using it execute the decrypt script
  - Example with Hugging Face
- Additional Training Examples [training data](m2qa_dataset/Additional_Training_data) come from the same datasets but the train split. Also uploaded on Hugging Face. And since its training data it is unencrypted.



The training data:

```bash
unzip -P m2qa german.zip
unzip -P m2qa chinese.zip
unzip -P m2qa turkish.zip
```

### Via Hugging Face

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
