# Dataset provenance and redistribution decision

## Release decision

The six speech files used for the benchmark are **not included in this public repository**. The
dataset was distributed as course material for UCSD CSE 256 in Fall 2024, but the supplied files and
assignment materials did not contain an explicit license or redistribution grant for the selected,
segmented, and labeled compilation.

The underlying text appears to consist of official speeches by Barack Obama, George W. Bush, and
George H. W. Bush. A spot check matches content in the archived White House record, and U.S.
Copyright Office guidance states that works prepared by federal employees as part of their official
duties generally are not protected by U.S. copyright under 17 U.S.C. §105:

- [U.S. Copyright Office, Title 17 §101 and §105](https://www.copyright.gov/title17/92chap1.html)
- [Archived Obama White House statement that WhiteHouse.gov content is public domain](https://obamawhitehouse.archives.gov/blog/2010/06/22/releasing-joint-strategic-plan-combat-intellectual-property-theft/)
- [Archived White House transcript matching a sampled Obama segment](https://obamawhitehouse.archives.gov/video/President-Obama-Speaks-to-the-Muslim-World-from-Cairo-Egypt?page=5)

That evidence supports the status of at least some underlying speeches, but it does not establish the
exact source and status of every passage or permission to redistribute UCSD's particular selection,
segmentation, labeling, and file compilation. Public availability is not itself a software/data
license. The conservative conclusion is therefore: **redistribution of this compiled dataset was not
verified, so it is excluded**.

This is a provenance audit, not legal advice.

## Expected local layout

Users who received an authorized copy can reproduce the experiments with:

```text
speechesdataset/
├── train_CLS.tsv
├── test_CLS.tsv
├── train_LM.txt
├── test_LM_obama.txt
├── test_LM_wbush.txt
└── test_LM_hbush.txt
```

Classification rows have the form `integer_label<TAB>speech segment`, with labels:

- `0`: Barack Obama
- `1`: George W. Bush
- `2`: George H. W. Bush

The CLI validates the required filenames and prints a direct reference to this document if any are
missing. Synthetic test fixtures under `tests/` exercise the code without the private data.

## Benchmark copy fingerprints

The August 14, 2026 release benchmark used the following local files. SHA-256 hashes allow an
authorized user to determine whether their copy is identical without publishing its contents.

| File | Records / tokens | SHA-256 |
|---|---:|---|
| `train_CLS.tsv` | 2,092 records | `7dbd7cadc6efe746d0c0a7e5fe81f0ac2431bd2a3b86b51f4cb23f373466c88e` |
| `test_CLS.tsv` | 750 records | `31277b7d67cff8ad717c9f43c05810c6b78feb67f04472bd7aa20e71eb2e0da8` |
| `train_LM.txt` | 31,896 tokens | `aff1d88fad90db8cedfe8918a9099ffdfe622641a2a8c6e25dac1f7eda199b1a` |
| `test_LM_obama.txt` | 5,461 tokens | `6cb3bb657ae40cab1cb8f3f38f391dbdcc80a9b21b577d5fc1ea4ba86901bed2` |
| `test_LM_wbush.txt` | 4,806 tokens | `1281773860075b12727f9f581940c028074d05f3efbf99bbfcc9c7c472d4dbd5` |
| `test_LM_hbush.txt` | 4,740 tokens | `a0592374b7cd47a6417a311892a4b8a59f35cbe054cbaf96c06e50e904c8451f` |

Token counts use this repository's lowercase word-and-punctuation tokenizer. The classification
training split contains 934 / 539 / 619 examples for labels 0 / 1 / 2; the test split is balanced at
250 examples per label.
