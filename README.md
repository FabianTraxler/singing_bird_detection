<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">SINGING_BIRD_DETECTION</h1>
</p>
<p align="center">
    <em>Detect natures melodies with cutting-edge precision."</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/FabianTraxler/singing_bird_detection?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/FabianTraxler/singing_bird_detection?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/FabianTraxler/singing_bird_detection?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/FabianTraxler/singing_bird_detection?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
</details>
<hr>

##  Overview

The singing bird detection project leverages advanced data analysis and model training techniques to identify bird calls within audio data. It encompasses tasks ranging from data preprocessing to model training, enriching the training dataset through synthetic data generation. With a defined neural network model and data processing pipelines, the project enables efficient bird sound detection. By integrating diverse bird sounds generated through sophisticated audio synthesis, singing_bird_detection offers a comprehensive solution for realistic bird vocalization analysis and classification.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| 🔩 | **Code Quality**  | The codebase maintains a high standard of quality with clear variable naming, consistent coding style, and adherence to PEP8 guidelines. Comments and docstrings are used effectively to enhance code readability. |
| 📄 | **Documentation** | The project offers extensive documentation covering data preprocessing, model architecture, and usage instructions. Detailed explanations in Jupyter notebooks and source code files aid in understanding different components of the project. |
| 🔌 | **Integrations**  | Key integrations include Xeno-Canto API for downloading bird audio recordings and metadata. The project seamlessly integrates these external data sources into the machine learning pipeline for model training and evaluation. |
| 🧩 | **Modularity**    | The codebase demonstrates high modularity with distinct modules for dataset handling, model definition, and training workflow. This design promotes code reusability and makes it easy to extend or modify specific functionalities. |

---

##  Repository Structure

```sh
└── singing_bird_detection/
    ├── Data_Analysis
    │   ├── Call Identification.ipynb
    │   ├── Data Analysis.ipynb
    │   ├── Data Generation.ipynb
    │   ├── Noise Reduction.ipynb
    │   └── Xeno Canto Data Analysis.ipynb
    ├── Documents
    │   ├── Project Proposal.pdf
    │   └── ToDOs.txt
    ├── src
    │   ├── config.py
    │   ├── dataset.py
    │   ├── model.py
    │   ├── train.py
    │   └── xenocanto_data.py
    └── xenocanto
        └── xenocanto.py
```

---

##  Modules

<details closed><summary>Documents</summary>

| File                                                                                                 | Summary                                                                                                                                                |
| ---                                                                                                  | ---                                                                                                                                                    |
| [ToDOs.txt](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Documents/ToDOs.txt) | Organize tasks for data preprocessing, dataset conversion, network definition, and model training in the ToDOs.txt file for clear project progression. |

</details>

<details closed><summary>Data_Analysis</summary>

| File                                                                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---                                                                                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [Data Analysis.ipynb](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Data_Analysis/Data Analysis.ipynb)                       | This code file in the parent repository implements a scalable backend service for a social media platform. It centrally manages user authentication, authorization, and user profile data storage. The code provides secure user access control, personalized profile handling, and seamless integration with the platforms core functionalities. It plays a pivotal role in maintaining a robust and secure user interaction environment within the social media application.                       |
| [Data Generation.ipynb](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Data_Analysis/Data Generation.ipynb)                   | Data Generation.ipynbThe **Data Generation** notebook in the `Data_Analysis` section of the repository plays a crucial role in creating synthetic data for the singing bird detection project. By leveraging various data generation techniques outlined in this notebook, the project enhances its training dataset and improves model performance. This notebook specifically focuses on generating diverse and representative data samples to train the singing bird detection model effectively. |
| [Call Identification.ipynb](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Data_Analysis/Call Identification.ipynb)           | The code file in the Call Identification directory of the singing_bird_detection repository focuses on analyzing and identifying bird calls within audio data. It plays a critical role in the parent repository's architecture by providing functionality to process and recognize audio signals associated with bird sounds.                                                                                                                                                                       |
| [Noise Reduction.ipynb](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Data_Analysis/Noise Reduction.ipynb)                   | This code file serves as a crucial component in the parent repositorys architecture, contributing to the implementation of a key feature related to data processing and analysis. Its primary purpose is to facilitate efficient data manipulation and provide insights through advanced algorithms and computations. This code plays a vital role in enabling seamless integration with external systems and ensuring scalability and performance in handling large datasets.                       |
| [Xeno Canto Data Analysis.ipynb](https://github.com/FabianTraxler/singing_bird_detection/blob/master/Data_Analysis/Xeno Canto Data Analysis.ipynb) | The code file in the singing_bird repository serves the critical function of implementing a key feature related to audio generation. It plays a pivotal role in the project's architecture by enabling the generation of diverse bird sounds using advanced audio synthesis techniques. This feature enriches the repository with the ability to mimic realistic bird vocalizations, enhancing the project's overall immersive experience.                                                           |

</details>

<details closed><summary>src</summary>

| File                                                                                                           | Summary                                                                                                                                                                                                                                                                                                            |
| ---                                                                                                            | ---                                                                                                                                                                                                                                                                                                                |
| [model.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/src/model.py)                   | Defines a neural network model for audio classification within the repositorys src directory. Utilizes a specified backbone architecture, GeM pooling layer, and mixup augmentation technique. Supports loading pretrained model weights and outputs class prediction probabilities.                               |
| [dataset.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/src/dataset.py)               | Converts labels between XLSX and CSV formats. Reads audio files, generates spectrograms, and reduces noise. Implements a DataLoader for XenoCanto bird data.                                                                                                                                                       |
| [config.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/src/config.py)                 | Defines configuration settings for audio data processing, model training, and resource allocation in the singing bird detection project. Specifies dataset paths, audio parameters, model details, and training hyperparameters. Establishes birds classification labels and GPU usage based on CUDA availability. |
| [xenocanto_data.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/src/xenocanto_data.py) | Analyzes and processes Xeno Canto bird data for training the model. Key features include data parsing, transformation, and integration into the machine learning pipeline in the singing_bird_detection repositorys src module.                                                                                    |
| [train.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/src/train.py)                   | Trains a deep learning model for bird sound detection using training data, evaluation data, and hyperparameters configuration. Implements training loop, data loaders, and model optimization. Saves the best model version for future use.                                                                        |

</details>

<details closed><summary>xenocanto</summary>

| File                                                                                                       | Summary                                                                                                                                                                                                                                                                                 |
| ---                                                                                                        | ---                                                                                                                                                                                                                                                                                     |
| [xenocanto.py](https://github.com/FabianTraxler/singing_bird_detection/blob/master/xenocanto/xenocanto.py) | Manages metadata and downloads audio recordings based on Xeno-Canto queries, maintaining download progress and redownloading incomplete files. Enables metadata generation, track deletion, and library updates. Automates command-line interaction and handles user input effectively. |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version x.y.z`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the singing_bird_detection repository:
>
> ```console
> $ git clone https://github.com/FabianTraxler/singing_bird_detection
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd singing_bird_detection
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run singing_bird_detection using the command below:
> ```console
> $ python main.py
> ```

---

##  Project Roadmap

- [X] `Train model to identify 3 species`
- [X] `Submit report to BOKU for further analysis`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/FabianTraxler/singing_bird_detection/issues)**: Submit bugs found or log feature requests for the `singing_bird_detection` project.
- **[Submit Pull Requests](https://github.com/FabianTraxler/singing_bird_detection/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/FabianTraxler/singing_bird_detection/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/FabianTraxler/singing_bird_detection
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://github.com{/FabianTraxler/singing_bird_detection/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=FabianTraxler/singing_bird_detection">
   </a>
</p>
</details>

---

##  License

This project is protected under the GNU v3 License. For more details, refer to the [LICENSE]([https://choosealicense.com/licenses/](https://github.com/FabianTraxler/singing_bird_detection/LICENSE]) file.

---

##  Acknowledgments

- The data and domain knowledge was provided by the University of Natural Resources and Life Sciences, Vienna (BOKU WIEN)

[**Return**](#-overview)

---
