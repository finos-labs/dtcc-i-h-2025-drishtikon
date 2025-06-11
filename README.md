![badge-labs](https://user-images.githubusercontent.com/327285/230928932-7c75f8ed-e57b-41db-9fb7-a292a13a1e58.svg)

# DTCC AI Hackathon 2025: Empowering India's Innovators
The purpose of hackathon is to leverage AI and ML Technologies to address critical challenges in the financial markets. The overall goal is to progress industry through Innovation, Networking and by providing effective Solutions.

**Hackathon Key Dates** 
•	June 6th - Event invites will be sent to participants
•	June 9th - Hackathon Open
•	June 9th-11th - Team collaboration and Use Case development
•	June 12th - Team presentations & demos
•	June 16th - Winners Announcement

More Info - https://communications.dtcc.com/dtcc-ai-hackathon-registration-17810.html

Commit Early & Commit Often!!!

# 🚀 Investica
### *AI-Powered Financial Assistant for Next-Gen Trading*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![AI Powered](https://img.shields.io/badge/AI-Powered-purple?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

**Revolutionizing portfolio management through intelligent AI agents**

[🎯 Live Demo](#-demo) • [📖 Documentation](#-table-of-contents) • [🛠️ Setup](#-quick-start) • [🤝 Contribute](#-contributing)

</div>

---

## 🌟 Overview

**Investica** is a cutting-edge AI-powered financial assistant that transforms how retail investors approach portfolio management and trading. Built for the **DTCC i-Hack 2025 - Drishtikon**, our platform leverages advanced agentic AI workflows to deliver seamless integration between Indian markets (via Zerodha Kite API) and global financial insights.

### 🎯 What Makes Investica Special?

```mermaid
graph TD
    A[👤 User Query] --> B[🧠 AI Routing Agent]
    B --> C[📊 Portfolio Assessment]
    B --> D[⚡ Trade Execution]
    B --> E[📋 Post-Trade Processing]
    C --> F[💼 Smart Recommendations]
    D --> F
    E --> F
    F --> G[📱 Beautiful UI Response]
```

---

## ✨ Core Features

<table>
<tr>
<td width="50%">

### 🔍 **Intelligent Portfolio Analysis**
- **Real-time Assessment**: Live portfolio health monitoring
- **Multi-source Data**: Combines Zerodha + global financial data
- **AI Insights**: Advanced recommendation engine
- **Risk Analytics**: Comprehensive risk profiling

</td>
<td width="50%">

### ⚡ **Smart Trade Execution**
- **Market Optimization**: Real-time condition analysis
- **Automated Trading**: GTT and conditional orders
- **Multi-asset Support**: Stocks, derivatives, and more
- **Performance Tracking**: Live P&L monitoring

</td>
</tr>
<tr>
<td width="50%">

### 🤖 **Agentic AI Workflow**
- **7 Specialized Agents**: Each optimized for specific tasks
- **Natural Language**: Chat-based interface
- **Memory Persistence**: Context-aware conversations
- **Workflow Orchestration**: Seamless agent collaboration

</td>
<td width="50%">

### 🌐 **Global Market Intelligence**
- **Indian Markets**: Full Zerodha Kite integration
- **US Markets**: Comprehensive financial datasets
- **Crypto Insights**: Digital asset analysis
- **News Integration**: Real-time market sentiment

</td>
</tr>
</table>

---

## 🏗️ Architecture Deep Dive

### 🤖 Meet the AI Agents

<div align="center">

| Agent | Role | Technology | Key Features |
|-------|------|------------|-------------|
| 🧭 **Routing Agent** | Query orchestration | Claude + AWS Bedrock | NLP parsing, task distribution |
| 📊 **Portfolio Assessor** | Investment analysis | Pydantic AI | Risk assessment, recommendations |
| 🎯 **Trade Executor** | Order management | LangGraph | Smart execution, optimization |
| 📋 **Post-Trade Manager** | Settlement processing | MCP Protocol | Reconciliation, reporting |
| 🌍 **Financial Analyst** | Global data analysis | Financial Datasets API | Fundamental analysis, insights |
| 💹 **Zerodha Agent** | Indian market interface | Kite API | Trading, portfolio data |
| 📤 **Output Agent** | Response synthesis | Streamlit | Visualization, user experience |

</div>

### 🔄 Workflow Architecture

![Workflow Architecture](https://github.com/user-attachments/assets/97b6f2a3-5c2a-4f0f-bd2b-14d18af3c040)


## 🛠️ Technology Stack

<div align="center">

### **Agentic AI Infrastructure**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic_AI-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3A3A?style=for-the-badge&logo=langchain&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-000000?style=for-the-badge&logo=anthropic&logoColor=white)
![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![MCP](https://img.shields.io/badge/MCP_Protocol-4A90E2?style=for-the-badge&logo=protocol&logoColor=white)

### **Frontend & UI**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### **Data & APIs**
![Zerodha](https://img.shields.io/badge/Zerodha_Kite-387ED1?style=for-the-badge&logo=zerodha&logoColor=white)
![SupaBase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Financial APIs](https://img.shields.io/badge/Financial_Datasets-2E8B57?style=for-the-badge&logo=chartdotjs&logoColor=white)

</div>

---

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.11+** 🐍
- **Git** 📚
- **AWS Bedrock Access** ☁️
- **SupaBase Account** 🗄️

### 🔑 Required API Keys

| Service | Purpose | Get From |
|---------|---------|----------|
| 🏦 Zerodha Kite | Indian market trading | [kite.trade](https://kite.trade) |
| 📈 Financial Datasets | Global market data | [financialdatasets.ai](https://financialdatasets.ai) |
| 🤖 AWS Bedrock | Claude AI access | [AWS Console](https://aws.amazon.com/bedrock) |
| 🗄️ SupaBase | Database persistence | [supabase.com](https://supabase.com) |

### ⚡ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/investica.git
   cd investica
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Launch the application**
   ```bash
   streamlit run app/main.py
   ```

5. **Access your dashboard**
   ```
   🌐 http://localhost:8501
   ```

---

## 💡 Usage Examples

### 🎯 Sample Queries

<table>
<tr>
<td width="50%">

**📊 Portfolio Analysis**
```
"Analyze my current portfolio 
and suggest optimization strategies"
```

**📈 Investment Research**
```
"Research AAPL fundamentals and 
compare with Indian tech stocks"
```

</td>
<td width="50%">

**⚡ Smart Trading**
```
"Buy 100 RELIANCE shares if 
price drops below ₹2900"
```

**📋 Trade Management**
```
"Show me today's trade summary 
with P&L breakdown"
```

</td>
</tr>
</table>

### 📊 Sample Output

```markdown
## 📊 Portfolio Health Report

| 🏢 Stock | 📊 Qty | 💰 Avg Price | 📈 Current | 🎯 Action |
|----------|---------|--------------|------------|-----------|
| RELIANCE | 100     | ₹2,950      | ₹3,000     | 🟢 **HOLD** |
| TCS      | 50      | ₹4,100      | ₹4,150     | 🔴 **SELL** (Overvalued) |
| INFY     | 75      | ₹1,650      | ₹1,700     | 🟡 **MONITOR** |

### 🌍 Global Insights - AAPL
- **📈 Revenue (2024)**: $390.1B – Steady growth trajectory
- **💡 Recommendation**: Consider adding for portfolio diversification
- **🎯 Target Price**: $195 (Current: $185)

### ⚡ Automated Actions Taken
✅ Placed GTT order: Sell 50 TCS @ ₹4,200  
✅ Set price alert: AAPL below $180  
✅ Updated risk profile: Moderate → Balanced  
```

---

## 🎨 Screenshots

![Dashboard](https://github.com/user-attachments/assets/b1689a5a-24c8-4eb6-9e97-25235fd9a74b)

![Thinking](https://github.com/user-attachments/assets/422f94e9-1d39-42f7-9f52-ed1b9cf27b74)

![Response](https://github.com/user-attachments/assets/e8dad51b-4007-4f3d-b1f9-edd5a3766832)

![buying stock](https://github.com/user-attachments/assets/7fafb759-3d2b-44d5-95b0-920324050805)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🌟 Ways to Contribute

<table>
<tr>
<td width="25%">

**🐛 Bug Reports**
- Found an issue?
- Submit detailed reports
- Help us improve!

</td>
<td width="25%">

**✨ Feature Requests**
- Have ideas?
- Share your vision
- Shape the future!

</td>
<td width="25%">

**💻 Code Contributions**
- Fork & PR
- Follow guidelines
- Build together!

</td>
<td width="25%">

**📖 Documentation**
- Improve docs
- Add examples
- Help others learn!

</td>
</tr>
</table>

### 🔄 Development Workflow

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### 📋 Code Standards

- ✅ Follow **PEP 8** style guidelines
- ✅ Add **comprehensive tests**
- ✅ Update **documentation**
- ✅ Use **meaningful commit messages**

---

## 📊 Project Stats

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/your-username/investica?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/investica?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/investica)
![GitHub PRs](https://img.shields.io/github/issues-pr/your-username/investica)

</div>

---

## 🏆 Achievements

<div align="center">

🥇 **DTCC i-Hack 2025 - Drishtikon Finalist**  
🚀 **Most Innovative AI Solution**  
⭐ **Best User Experience Design**  
💡 **Outstanding Technical Implementation**

</div>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DTCC i-Hack 2025** for the amazing opportunity
- **Zerodha** for their excellent Kite API
- **Anthropic** for Claude AI capabilities
- **Open Source Community** for incredible tools and libraries

---

<div align="center">

## 🚀 **Ready to Transform Your Trading?**

### [🎯 Try Investica Now](http://localhost:8501) | [📖 Read the Docs](#) | [💬 Join Community](#)

**Built with ❤️ for the future of financial technology**

---

*Investica - Where AI meets smart investing* ✨

![Footer](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)
![AI](https://img.shields.io/badge/Powered%20by-AI-blue?style=for-the-badge)
![Innovation](https://img.shields.io/badge/Innovation-First-purple?style=for-the-badge)

</div>

### Project Details


### Team Information


## Using DCO to sign your commits

**All commits** must be signed with a DCO signature to avoid being flagged by the DCO Bot. This means that your commit log message must contain a line that looks like the following one, with your actual name and email address:

```
Signed-off-by: John Doe <john.doe@example.com>
```

Adding the `-s` flag to your `git commit` will add that line automatically. You can also add it manually as part of your commit log message or add it afterwards with `git commit --amend -s`.

See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for more information

### Helpful DCO Resources
- [Git Tools - Signing Your Work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [Signing commits
](https://docs.github.com/en/github/authenticating-to-github/signing-commits)


## License

Copyright 2025 FINOS

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)








