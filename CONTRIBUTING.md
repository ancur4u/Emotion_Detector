# Contributing to Emotion Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/ancur4u/Emotion_Detector.git
   cd emotion-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

## 🎯 How to Contribute

### Reporting Bugs

1. **Check existing issues** first
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. **Check if feature already exists** or is planned
2. **Create an issue** with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach

### Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   flake8 emotion_app.py improved_trainer.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **meaningful variable names**
- Add **docstrings** for functions and classes
- Keep **functions small** and focused
- Use **type hints** where appropriate

### Code Quality

```python
# Good example
def extract_facial_features(image: np.ndarray) -> np.ndarray:
    """Extract facial features from preprocessed image.
    
    Args:
        image: Grayscale image array of shape (48, 48)
        
    Returns:
        Feature vector of extracted facial characteristics
    """
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale")
    
    features = []
    # Implementation here...
    return np.array(features)
```

### Testing

- Write **unit tests** for new functions
- Ensure **test coverage** for critical paths
- Use **descriptive test names**

```python
def test_feature_extraction_with_valid_image():
    """Test that feature extraction works with valid 48x48 image."""
    detector = EmotionDetector()
    test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    
    features = detector.extract_features(test_image)
    
    assert len(features) == 2343  # Expected feature count
    assert np.all(features >= 0)  # All features should be non-negative
```

## 🔧 Development Guidelines

### Adding New Features

1. **Feature extraction improvements**
   - Add new feature types in `enhanced_feature_extraction()`
   - Update feature count documentation
   - Test with existing models

2. **Model architecture changes**
   - Maintain backward compatibility
   - Update model saving/loading logic
   - Test with different model types

3. **UI improvements**
   - Follow Streamlit best practices
   - Ensure responsive design
   - Add appropriate error handling

### Performance Considerations

- **Optimize for inference speed** (real-time detection)
- **Memory efficiency** for large datasets
- **Scalable feature extraction** for different image sizes
- **Model size** considerations for deployment

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_emotion_detector.py

# Run with coverage
python -m pytest tests/ --cov=emotion_app --cov-report=html
```

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: Component interaction testing
3. **Performance Tests**: Speed and memory usage
4. **UI Tests**: Streamlit interface testing

## 📚 Documentation

### Code Documentation

- Add **docstrings** to all public functions
- Update **README.md** for new features
- Include **usage examples** where helpful

### API Documentation

```python
def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
    """Predict emotion from face image.
    
    Args:
        face_img: RGB or grayscale face image
        
    Returns:
        Tuple of (emotion_name, confidence_score)
        
    Raises:
        ValueError: If no model is loaded
        RuntimeError: If feature extraction fails
    """
```

## 🎨 Commit Guidelines

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or modifying tests
- **perf**: Performance improvements

### Examples

```bash
git commit -m "feat(detector): add LBP texture features for better accuracy"
git commit -m "fix(ui): resolve camera permission error on mobile"
git commit -m "docs(readme): update installation instructions"
```

## 🔍 Code Review Process

### Before Submitting PR

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Feature works in Streamlit app

### Review Criteria

1. **Functionality**: Does it work as intended?
2. **Code Quality**: Is it readable and maintainable?
3. **Performance**: Does it impact speed or memory?
4. **Testing**: Are there adequate tests?
5. **Documentation**: Is it properly documented?

## 🐛 Issue Labels

- **bug**: Something isn't working
- **enhancement**: New feature request
- **documentation**: Documentation improvements
- **good first issue**: Good for newcomers
- **help wanted**: Community help needed
- **performance**: Performance improvements
- **ui/ux**: User interface improvements

## 🎉 Recognition

Contributors will be:
- **Listed in README.md**
- **Mentioned in release notes**
- **Invited to maintainer team** (for significant contributions)

## ❓ Questions?

- **Create an issue** for questions about contributing
- **Start a discussion** for broader topics
- **Contact maintainers** directly for urgent matters

Thank you for contributing! 🙏
