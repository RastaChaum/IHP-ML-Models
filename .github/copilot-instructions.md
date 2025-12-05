# GitHub Copilot Instructions - IHP ML Models

## 🎯 Project Overview

The IHP ML Models project is responsible for developing, training, and serving machine learning models used by the Intelligent Heating Pilot (IHP) Home Assistant integration. This document defines the architectural principles and development practices that **must** be followed by all AI-assisted code generation.

## 🛡️ Architectural Mandate: Domain-Driven Design (DDD)

All development must follow **Domain-Driven Design** principles with strict separation of concerns.

### Layer Structure

```
ihp_ml_addon/
├── domain/              # Pure business logic (NO external framework dependencies, e.g., scikit-learn, TensorFlow)
│   ├── value_objects/   # Immutable data carriers (e.g., TrainingData, ModelHyperparameters)
│   ├── entities/        # Domain entities and aggregates (e.g., MLModel, ModelTrainer)
│   ├── interfaces/      # Abstract base classes (contracts for data sources, model storage, prediction serving)
│   └── services/        # Domain services (e.g., ModelEvaluationService)
├── infrastructure/      # ML Framework & Data Integration Layer
│   ├── adapters/        # Implementations using ML libraries (e.g., ScikitLearnAdapter, DatabaseAdapter)
│   └── repositories/    # Data persistence and model storage implementations
└── application/         # Orchestration and use cases (e.g., TrainModelUseCase, PredictTemperatureUseCase)
```

### Domain Layer Rules (⚠️ CRITICAL)

The **domain layer** contains the core intellectual property and must be completely isolated:

1. **NO direct ML framework imports** - Zero `sklearn.*`, `tensorflow.*`, `pandas.*`, `numpy.*` imports allowed directly. Interactions must be via interfaces.
2. **NO external service dependencies** - Only Python standard library and domain code.
3. **Pure business logic** - If it models real-world ML behavior, feature engineering concepts, or model evaluation, it belongs here.
4. **Interface-driven** - All external interactions (data loading, model training, prediction) via Abstract Base Classes (ABCs).
5. **Type hints required** - All functions and methods must have complete type annotations.

### Infrastructure Layer Rules

The **infrastructure layer** bridges the domain to ML frameworks, data sources, and deployment environments:

1. **Implements domain interfaces** - All adapters must implement ABCs from domain layer.
2. **ML framework specific code only** - All `sklearn.*`, `tensorflow.*`, `pandas.*`, `numpy.*` imports belong here.
3. **Thin adapters** - Minimal logic, just translation between ML frameworks/data and domain.
4. **No business logic** - Delegate all decisions to domain layer.

## 🧪 Test-Driven Development (TDD) Standard

All new features must be developed using TDD:

### Unit Testing Requirements

1. **Domain-first testing** - Write domain layer tests BEFORE implementation.
2. **Mock external dependencies** - Use mocks for all infrastructure interactions (ML frameworks, data sources).
3. **Test against interfaces** - Unit tests should test against ABCs, not concrete implementations.
4. **Centralized fixtures** - Use a centralized `fixtures.py` file for test data (DRY principle).
5. **High coverage** - Aim for >80% coverage of domain logic.
6. **Fast tests** - Domain tests should run in milliseconds (no heavy ML computations, no I/O).

### Logging Standards

1. **Method Entry/Exit Logging** - All public methods in the domain and application layers must log at `INFO` level on entry and exit.
2. **Parameter/Return Value Logging** - Input parameters and return values should be logged at `DEBUG` level.
3. **Structured Logging** - Prefer structured logging (e.g., JSON) where possible for easier parsing and analysis.

### Testing Structure

```python
tests/
├── unit/
│   ├── domain/          # Pure domain logic tests (e.g., feature engineering rules, model evaluation logic)
│   │   ├── test_value_objects.py
│   │   ├── test_ml_model.py
│   │   └── test_domain_services.py
│   └── infrastructure/  # Adapter tests (with mocked ML frameworks/data sources)
│       ├── test_scikit_learn_adapter.py
│       └── test_data_repository.py
└── integration/         # Tests involving real ML framework usage or data sources (slower, e.g., train a small model)
└── e2e/                 # End-to-end tests (optional, slowest, e.g., full pipeline from data ingestion to prediction)
```

### Example: Testing with Interfaces

```python
# domain/interfaces/data_loader.py
from abc import ABC, abstractmethod
from domain.value_objects import TrainingData

class IDataLoader(ABC):
    """Contract for loading training data."""
    
    @abstractmethod
    async def load_data(self) -> TrainingData:
        """Load training data."""
        pass

# tests/unit/domain/test_model_trainer.py
from unittest.mock import Mock
from domain.interfaces.data_loader import IDataLoader
from domain.entities.model_trainer import ModelTrainer
from domain.value_objects import TrainingData, ModelHyperparameters

def test_model_trainer_trains_model():
    # GIVEN: Mock data loader and hyperparameters
    mock_data_loader = Mock(spec=IDataLoader)
    mock_data_loader.load_data.return_value = TrainingData(...) # Populate with dummy data
    hyperparameters = ModelHyperparameters(...) # Populate with dummy hyperparameters
    
    # WHEN: Trainer trains a model
    trainer = ModelTrainer(data_loader=mock_data_loader)
    model = trainer.train_model(hyperparameters)
    
    # THEN: Should return a trained model
    assert model is not None # Or assert specific model properties
```

## 🎯 Initial Implementation: Core Abstractions

### A. Value Objects (Immutable Data Carriers)

Use Python **dataclasses** with `frozen=True` for all value objects:

```python
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass(frozen=True)
class TrainingData:
    """Input data for model training."""
    features: pd.DataFrame
    target: pd.Series
    timestamp: datetime

@dataclass(frozen=True)
class ModelHyperparameters:
    """Configuration parameters for an ML model."""
    learning_rate: float
    n_estimators: int
    random_state: int

@dataclass(frozen=True)
class ModelPrediction:
    """Result of a model prediction."""
    predicted_value: float
    confidence_score: float
    timestamp: datetime
```

### B. The ML Model (Aggregate Root)

```python
from domain.interfaces.model_trainer import IModelTrainer
from domain.interfaces.model_storage import IModelStorage
from domain.value_objects import ModelPrediction, ModelHyperparameters, TrainingData

class MLModel:
    """Represents a trained machine learning model."""
    
    def __init__(
        self,
        model_trainer: IModelTrainer,
        model_storage: IModelStorage,
    ) -> None:
        self._trainer = model_trainer
        self._storage = model_storage
        self._trained_model = None # Placeholder for the actual trained model instance
    
    async def train_and_save(
        self, 
        training_data: TrainingData, 
        hyperparameters: ModelHyperparameters
    ) -> None:
        """Trains the model with new data and saves it."""
        self._trained_model = await self._trainer.train_model(training_data, hyperparameters)
        await self._storage.save_model(self._trained_model)

    async def predict(self, input_features: pd.DataFrame) -> ModelPrediction:
        """Generates a prediction using the loaded model."""
        if not self._trained_model:
            raise ValueError("Model not trained or loaded.")
        # Pure business logic here - no direct ML framework dependencies
        pass
```

### C. Interface Contracts (ABCs)

Define clear contracts for all external interactions:

```python
# domain/interfaces/data_loader.py
from abc import ABC, abstractmethod
from domain.value_objects import TrainingData

class IDataLoader(ABC):
    """Contract for loading training data."""
    
    @abstractmethod
    async def load_data(self) -> TrainingData:
        """Load training data."""
        pass

# domain/interfaces/model_trainer.py
from abc import ABC, abstractmethod
from domain.value_objects import TrainingData, ModelHyperparameters

class IModelTrainer(ABC):
    """Contract for training machine learning models."""
    
    @abstractmethod
    async def train_model(self, training_data: TrainingData, hyperparameters: ModelHyperparameters) -> object:
        """Train an ML model and return the trained model object."""
        pass

# domain/interfaces/model_storage.py
from abc import ABC, abstractmethod

class IModelStorage(ABC):
    """Contract for persisting and loading ML models."""
    
    @abstractmethod
    async def save_model(self, model: object) -> None:
        """Persist a trained ML model."""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> object:
        """Load a trained ML model."""
        pass

# domain/interfaces/prediction_service.py
import pandas as pd
from abc import ABC, abstractmethod
from domain.value_objects import ModelPrediction

class IPredictionService(ABC):
    """Contract for serving model predictions."""
    
    @abstractmethod
    async def get_prediction(self, input_features: pd.DataFrame) -> ModelPrediction:
        """Get a prediction from the loaded model."""
        pass
```

## 📝 Code Style & Quality Standards

### Type Hints

- **Always use type hints** for function parameters and return values
- Use `from __future__ import annotations` for forward references
- Use `typing` module types: `Optional`, `Union`, `Protocol`, etc.
- Use `None` instead of `Optional[T]` for Python 3.10+ (written as `T | None`)

### Documentation

- **Docstrings required** - For all public classes and methods
- **User-centric and Developer-centric Docs** - Maintain separate, clear documentation streams: one focused on the user experience (how to use the ML models), and another detailed guide for developers (API, architecture, contribution guidelines).
- **No Markdown Reports in Repository** - Avoid generating new Markdown files for reports (e.g., test reports, analysis summaries) directly within the repository. Such reports should be communicated in the pull request comments or directly in the conversation.
- Use Google-style docstrings
- Include type information in docstrings only when it adds clarity
- Document business rules and assumptions

### Code Organization

- **Single Responsibility** - Each class/function does one thing
- **Prefer Refactoring** - Prioritize refactoring existing classes and methods over creating new ones to avoid code bloat and ensure consistency.
- **Small functions** - Prefer functions under 20 lines
- **Clear naming** - Use descriptive names (e.g., `train_model` not `tm`)
- **No magic numbers** - Use named constants
- **DRY Principle** - Actively seek and eliminate code duplication, not just in test fixtures but across the entire codebase.

### Python Standards

- Follow **PEP 8** style guide
- Use **dataclasses** for simple data structures
- Prefer **composition over inheritance**
- Use **async/await** for I/O operations
- Avoid global state

## 🚫 Anti-Patterns to Avoid

1. ❌ **Tight coupling to ML frameworks in Domain Layer**
   ```python
   # BAD: Domain logic mixed with scikit-learn
   from sklearn.ensemble import RandomForestRegressor

   class ModelTrainer:
       def train(self, data):
           model = RandomForestRegressor()
           model.fit(data.features, data.target)
   ```
   
   ✅ **Good: Clean separation via interfaces**
   ```python
   # GOOD: Domain receives an interface
   from domain.interfaces.model_trainer import IModelTrainer

   class MLModel:
       def __init__(self, trainer: IModelTrainer):
           self._trainer = trainer
       
       def train_model(self, data, hyperparameters):
           model = self._trainer.train_model(data, hyperparameters)
   ```

2. ❌ **Business logic in infrastructure (e.g., in a Scikit-Learn adapter)**
   ```python
   # BAD: Decision-making in adapter
   class ScikitLearnAdapter:
       async def train_model(self, data, hyperparameters):
           # ... actual training ...
           if model.score < 0.8: # Business rule!
               raise ValueError("Model performance too low")
   ```
   
   ✅ **Good: Infrastructure only translates and executes framework calls**
   ```python
   # GOOD: Adapter just trains and returns the model
   class ScikitLearnAdapter:
       async def train_model(self, data, hyperparameters):
           model = RandomForestRegressor(**hyperparameters.to_dict())
           model.fit(data.features, data.target)
           return model # Just returns the trained model
   ```

3. ❌ **Untestable code (direct ML framework dependency)**
   ```python
   # BAD: Hard to test (direct scikit-learn dependency)
   def train_and_evaluate():
       from sklearn.ensemble import RandomForestRegressor
       model = RandomForestRegressor()
       # ...
   ```
   
   ✅ **Good: Testable with interfaces**
   ```python
   # GOOD: Easily mockable
   def train_and_evaluate(trainer: IModelTrainer, data: TrainingData):
       model = trainer.train_model(data, ModelHyperparameters(...))
       # ...
   ```

## 📚 Summary Checklist for New Code

Before submitting any AI-generated code, verify:

- [ ] Domain layer has NO direct ML framework imports (e.g., `sklearn`, `pandas`, `numpy`)
- [ ] All external interactions use ABCs (interfaces)
- [ ] Value objects are immutable (`@dataclass(frozen=True)`)
- [ ] All functions have complete type hints
- [ ] Unit tests exist and use mocks for dependencies
- [ ] Tests use centralized fixtures (DRY principle)
- [ ] Tests can run without heavy ML computations or I/O for unit tests
- [ ] Business logic is in domain, infrastructure is thin
- [ ] No hardcoded fallback values - prefer WARNING logs and user alerts
- [ ] Code follows PEP 8 and uses meaningful names
- [ ] Docstrings explain the "why", not just the "what"

## 🎓 Philosophy

> "The domain of our ML models is our valuable intellectual property. It must be protected from framework specifics, testable in isolation, and clear in its intent. When ML frameworks change, our domain logic remains stable. When we change our ML logic, tests catch regressions immediately."

**Focus on defining clear boundaries and interfaces first.** The quality of abstractions determines the quality of the entire system.
