# Python Installation & Setup Guide

Complete guide for setting up Python on Linux, fixing pip issues, and installing project dependencies.

---

## 1. Installing Python on Linux

### Step 1: Update Package Manager
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Python 3 (Latest Stable)
```bash
sudo apt install python3 python3-pip python3-venv build-essential -y
```

### Step 3: Verify Installation
```bash
python3 --version
pip3 --version
```

**Expected Output:**
```
Python 3.x.x
pip 23.x.x from /usr/lib/python3/dist-packages/pip (python 3.x)
```

---

## 2. Fix Failed Pip Install

The `pip install` command failed because it needs a package name or requirements file.

### Common Causes:
- **Missing requirements file path**
- **System-wide pip restrictions**
- **Missing build tools**

### Solution: Upgrade pip First
```bash
python3 -m pip install --upgrade pip setuptools wheel
```

---

## 3. Create Virtual Environment

**Why?** Virtual environments isolate project dependencies, preventing conflicts.

### Step 1: Navigate to Project Directory
```bash
cd /home/enock/Desktop/youth_unemployment_data_driven
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**Expected Output:** Your terminal prompt changes to:
```
(venv) user@machine:~/youth_unemployment_data_driven$
```

### Step 4: Verify Activation
```bash
which python
```

**Expected:** Points to `venv/bin/python`

---

## 4. Install Project Dependencies

### Option A: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Option B: Install Individual Packages (if needed)
```bash
pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn scipy openpyxl requests
```

### Verify Installation
```bash
pip list
```

You should see all packages from `requirements.txt` listed.

---

## 5. Troubleshooting

### Issue: `command not found: python3`
```bash
sudo apt install python3
```

### Issue: Permission Denied Error
```bash
# Don't use sudo with pip in virtual environments!
# Make sure your venv is activated:
source venv/bin/activate
```

### Issue: Slow pip Installation
```bash
# Try upgrading pip and use a faster mirror:
pip install --upgrade pip
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

### Issue: "ERROR: Could not find a version that satisfies the requirement"
```bash
# Update pip/setuptools and try again:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue: Build Errors (Missing C Compiler)
```bash
sudo apt install build-essential python3-dev
```

---

## 6. Deactivate Virtual Environment

When done working:
```bash
deactivate
```

---

## 7. Quick Reference Commands

| Command | Purpose |
|---------|---------|
| `python3 --version` | Check Python version |
| `pip3 --version` | Check pip version |
| `python3 -m venv venv` | Create virtual environment |
| `source venv/bin/activate` | Activate venv (Linux/Mac) |
| `deactivate` | Exit virtual environment |
| `pip install -r requirements.txt` | Install all dependencies |
| `pip list` | List installed packages |
| `pip freeze > requirements.txt` | Save current packages |

---

## 8. Running Your Project

### After Setup is Complete:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app (if applicable)
streamlit run home.py

# Or run Python scripts
python3 app_integration.py
```

---

## 9. Project Dependencies Summary

Your `requirements.txt` includes:

- **Streamlit**: Web app framework
- **Pandas/NumPy**: Data manipulation
- **Plotly/Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning
- **Requests**: HTTP requests
- **Statsmodels**: Statistical analysis

---

## 10. Additional Resources

- [Python Official Docs](https://docs.python.org/3/)
- [pip Documentation](https://pip.pypa.io/)
- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Last Updated:** February 25, 2026
