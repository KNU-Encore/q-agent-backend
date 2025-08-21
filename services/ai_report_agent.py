import json
import re
from typing import Dict, Any, List, Optional

import redis
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


class SimpleHallucinationDetector:
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", temperature: float = 0.1):
        self.llm = ChatAnthropic(temperature=temperature, model=model_name)

    def detect_hallucinations(self, report: str, original_data: dict) -> dict:
        try:
            context_data = f"""
            Original Query Data:
            Query: {original_data.get('query', 'N/A')}
            Execution Time: {original_data.get('performance_metrics', {}).get('qt', 'N/A')} seconds
            Lock Time: {original_data.get('performance_metrics', {}).get('lt', 'N/A')} seconds
            Rows Sent: {original_data.get('performance_metrics', {}).get('rsent', 'N/A')}
            Rows Examined: {original_data.get('performance_metrics', {}).get('rexp', 'N/A')}
            Execution Plan: {str(original_data.get('plan_tree', 'N/A'))[:1000]}
            Schema Info: {str(original_data.get('schema_info', 'N/A'))[:500]}
            """

            verification_prompt = f"""
            You are a MySQL-specific SQL performance analysis fact-checker. Your task is to verify the accuracy of a MySQL query performance analysis report against the original query data.

            **MYSQL DATABASE CONTEXT**: This analysis is specifically for MySQL database queries. Verify that all technical recommendations and syntax are MySQL-compatible.

            ORIGINAL DATA:
            {context_data}

            ANALYSIS REPORT TO VERIFY:
            {report}

            Please carefully review the analysis report and verify its accuracy against the original data. Focus on:

            1. **Factual Claims Verification:**
               - Are the performance metrics (execution time, rows sent/examined) correctly stated?
               - Are the technical statements about MySQL operations accurate?
               - Do the optimization recommendations align with MySQL capabilities and the actual query structure?
               - Are any numerical values or calculations correct?

            2. **MySQL-Specific Technical Accuracy:**
               - Are execution plan interpretations consistent with MySQL execution plans?
               - Are the identified bottlenecks supported by the actual MySQL query data?
               - Are index recommendations appropriate for MySQL and the query structure?
               - Are join algorithm assessments accurate for MySQL's join implementations?
               - Are storage engine considerations (InnoDB/MyISAM) properly addressed?

            3. **Consistency Check:**
               - Does the analysis stay within the bounds of the provided data?
               - Are conclusions logically derived from the available MySQL-specific information?
               - Are there any unsupported claims or speculative statements?
               - Are all syntax recommendations valid for MySQL?

            Please provide your assessment in the following format:

            **RELIABILITY SCORE:** [Score from 1-10, where 10 is completely accurate]

            **CONFIDENCE LEVEL:** [High/Medium/Low]

            **VERIFIED FACTS:** [Number of facts that were verified as correct]

            **QUESTIONABLE CLAIMS:** [Number of claims that seem incorrect or unsupported]

            **DETAILED ASSESSMENT:**
            [Provide a detailed explanation of your findings, including:]
            - What claims were verified as accurate
            - What claims appear questionable or incorrect
            - Any unsupported statements or potential hallucinations
            - Overall assessment of the report's reliability
            - MySQL-specific accuracy verification

            **SPECIFIC ISSUES FOUND:**
            [List any specific factual errors, inconsistencies, or unsupported claims you identified]
            """

            response = self.llm.invoke([HumanMessage(content=verification_prompt)])

            content = response.content

            reliability_score = 5
            score_match = re.search(
                r"RELIABILITY SCORE.*?(\d+(?:\.\d+)?)", content, re.IGNORECASE
            )
            if score_match:
                try:
                    reliability_score = float(score_match.group(1))
                except ValueError:
                    reliability_score = 5

            confidence_level = "Medium"
            confidence_match = re.search(
                r"CONFIDENCE LEVEL.*?(High|Medium|Low)", content, re.IGNORECASE
            )
            if confidence_match:
                confidence_level = confidence_match.group(1).title()

            verified_facts = 0
            verified_match = re.search(
                r"VERIFIED FACTS.*?(\d+)", content, re.IGNORECASE
            )
            if verified_match:
                try:
                    verified_facts = int(verified_match.group(1))
                except ValueError:
                    verified_facts = 0

            questionable_claims = 0
            questionable_match = re.search(
                r"QUESTIONABLE CLAIMS.*?(\d+)", content, re.IGNORECASE
            )
            if questionable_match:
                try:
                    questionable_claims = int(questionable_match.group(1))
                except ValueError:
                    questionable_claims = 0

            assessment_match = re.search(
                r"DETAILED ASSESSMENT:\s*\n(.*?)(?=\*\*SPECIFIC ISSUES|$)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            detailed_analysis = (
                assessment_match.group(1).strip()
                if assessment_match
                else "No detailed assessment provided"
            )

            issues_match = re.search(
                r"SPECIFIC ISSUES FOUND:\s*\n(.*?)$", content, re.IGNORECASE | re.DOTALL
            )
            questionable_statements = (
                issues_match.group(1).strip()
                if issues_match
                else "No specific issues identified"
            )

            if (
                questionable_claims > 0
                and questionable_statements == "No specific issues identified"
            ):
                questionable_statements = f"Analysis detected {questionable_claims} potentially questionable claims. Review recommended."

            return {
                "reliability_score": round(reliability_score, 1),
                "confidence_level": confidence_level,
                "total_claims": verified_facts + questionable_claims,
                "verified_claims": verified_facts,
                "questionable_claims": questionable_claims,
                "detailed_analysis": detailed_analysis,
                "questionable_statements": questionable_statements,
                "methodology": "Single comprehensive prompt-based verification",
                "full_response": content,
            }

        except Exception as e:
            print(f"Error in hallucination detection: {str(e)}")
            return {
                "reliability_score": 0,
                "confidence_level": "Unknown",
                "total_claims": 0,
                "verified_claims": 0,
                "questionable_claims": 0,
                "detailed_analysis": f"Error occurred during verification: {str(e)}",
                "questionable_statements": "Verification failed due to error",
                "methodology": "Single prompt verification (failed)",
                "error": str(e),
            }


class SummaryAgent:
    def __init__(self, llm):
        self.llm = llm

    def create_summary(self, performance_report: str, query_rewrite: str) -> str:
        try:
            alternative_1 = self._extract_alternative_approach(
                query_rewrite, "Approach 1"
            )
            alternative_2 = self._extract_alternative_approach(
                query_rewrite, "Approach 2"
            )
            index_recommendations = self._extract_index_recommendations(query_rewrite)

            summary_prompt = f"""
            You are an expert MySQL database performance analyst. Create a comprehensive summary that includes MySQL-specific analysis and recommendations.

            **IMPORTANT**: This is for MySQL database analysis. Ensure all recommendations and syntax are MySQL-compatible.

            1. A concise executive summary of the MySQL performance analysis
            2. The complete alternative MySQL query approaches from the optimization recommendations

            PERFORMANCE REPORT TO SUMMARIZE:
            {performance_report[:3000]}...

            Create a summary with the following structure:

            ## 📋 Executive Summary
            [3-5 bullet points covering:]
            - Current MySQL performance status (execution time and efficiency)
            - Main MySQL-specific performance bottlenecks identified
            - Key MySQL optimization opportunities
            - Expected improvement potential with MySQL optimizations
            - Priority MySQL-specific recommendations

            ## 🔄 Alternative MySQL Query Approaches
            {alternative_1 if alternative_1 else "### Alternative approaches will be preserved from original optimization report"}

            {alternative_2 if alternative_2 else ""}

            ## 📐 MySQL Index Recommendations
            {index_recommendations if index_recommendations else "### Index recommendations will be preserved from original optimization report"}

            Format using proper Markdown with bullet points, code blocks, and keep it informative but concise.
            Ensure all SQL syntax and recommendations are MySQL-specific.
            """

            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary_content = response.content

            if not alternative_1 and not alternative_2:
                summary_content += (
                    f"\n\n{self._extract_full_query_rewrite_section(query_rewrite)}"
                )

            return summary_content

        except Exception as e:
            return (
                f"## 📋 Executive Summary\n\n❌ **Error generating summary:** {str(e)}"
            )

    def _extract_alternative_approach(
        self, query_rewrite: str, approach_name: str
    ) -> str:
        try:
            pattern = (
                rf"(\*\*{approach_name}\*\*.*?)(?=\*\*Approach [2-9]|\*\*[^A]|\n###|\Z)"
            )
            match = re.search(pattern, query_rewrite, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return ""
        except:
            return ""

    def _extract_index_recommendations(self, query_rewrite: str) -> str:
        try:
            pattern = r"(###.*?Index Recommendations.*?)(?=###|\Z)"
            match = re.search(pattern, query_rewrite, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return ""
        except:
            return ""

    def _extract_full_query_rewrite_section(self, query_rewrite: str) -> str:
        try:
            pattern = r"(### 🔄 Alternative Query Approaches.*?)(?=### 📊 Performance Expectations|### ⚖️ Trade-off Analysis|\Z)"
            match = re.search(pattern, query_rewrite, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

            pattern = r"(.*Alternative.*Approaches.*?)(?=### [^A]|\Z)"
            match = re.search(pattern, query_rewrite, re.DOTALL | re.IGNORECASE)
            if match:
                return f"### 🔄 Alternative Query Approaches\n{match.group(1).strip()}"

            return ""
        except:
            return ""


class AnalysisState(TypedDict):
    query: str
    plan_tree: str
    performance_metrics: Dict[str, Any]
    schema_info: Dict[str, Any]
    efficiency: float
    missing_metrics: List[str]
    data_availability_note: str
    formatted_schema: str
    analysis_context: str
    performance_report: Optional[str]
    query_rewrite: Optional[str]
    summary_report: Optional[str]
    verified_report: Optional[str]
    error: Optional[str]
    data_quality: str
    schema_available: bool
    hallucination_report: Optional[Dict[str, Any]]
    fact_check_results: Optional[Dict[str, Any]]
    verification_context: Optional[str]


class SQLQueryAnalyzer:
    def __init__(self):
        self.llm = None
        self.hallucination_detector = None
        self.summary_agent = None
        self.vectorstore = None
        self.report_agent_executor = None
        self.rewrite_agent_executor = None
        self.summarize_report_agent_executor = None
        self.verification_agent_executor = None
        self.db_schema = None
        self.graph = None
        self.model_name = "claude-sonnet-4-20250514"
        self.initialize_components()

    def safe_numeric_value(self, value, default=0):
        if value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                return value
            return float(value) if value != "" else default
        except (ValueError, TypeError):
            return default

    def safe_division(self, numerator, denominator, default=0):
        num = self.safe_numeric_value(numerator, 0)
        den = self.safe_numeric_value(denominator, 1)

        if den == 0:
            return default
        return num / den

    def format_metric_display(self, value, default_display="N/A"):
        if value is None:
            return default_display
        try:
            if isinstance(value, (int, float)):
                if value == int(value):
                    return f"{int(value):,}"
                else:
                    return f"{value:.3f}"
            return str(value)
        except:
            return default_display

    def initialize_components(self):
        try:
            load_dotenv()

            temperature = 0.1

            self.llm = ChatAnthropic(temperature=temperature, model=self.model_name)

            self.hallucination_detector = SimpleHallucinationDetector()
            self.summary_agent = SummaryAgent(self.llm)
            self.setup_agents()
            self.setup_graph()

            return True

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False

    def load_query_data(self, json_file_path: str) -> dict:
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None

    def format_schema_info(self, schema_info):
        if not schema_info:
            return "### MySQL Database Schema: Not Available\n⚠️ Analysis will be limited without schema information."

        try:
            if isinstance(schema_info, dict):
                formatted_schema = "### MySQL Database Schema Information:\n"
                for key, value in schema_info.items():
                    formatted_schema += f"**{key}**: {value}\n"
                return formatted_schema
            elif isinstance(schema_info, str):
                return f"### MySQL Database Schema:\n{schema_info}"
            else:
                return f"### MySQL Database Schema:\n```json\n{json.dumps(schema_info, indent=2)}\n```"
        except Exception as e:
            return f"### MySQL Database Schema: Error formatting schema - {str(e)}"

    def setup_agents(self):
        try:
            tools = [
                Tool(
                    name="analyze_execution_plan",
                    description="Analyze a MySQL execution plan tree in detail",
                    func=self.analyze_execution_plan_detailed,
                )
            ]

            react_prompt_template = """            
            You are a helpful assistant. Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
            """

            react_prompt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                template=react_prompt_template,
            )

            report_agent = create_react_agent(self.llm, tools, react_prompt)
            self.report_agent_executor = AgentExecutor(
                agent=report_agent,
                tools=tools,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True,
            )

            rewrite_agent = create_react_agent(self.llm, tools, react_prompt)
            self.rewrite_agent_executor = AgentExecutor(
                agent=rewrite_agent,
                tools=tools,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True,
            )

            verification_agent = create_react_agent(self.llm, tools, react_prompt)
            self.verification_agent_executor = AgentExecutor(
                agent=verification_agent,
                tools=tools,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True,
            )

        except Exception as e:
            print(f"Agent setup error: {str(e)}")
            import traceback

            traceback.print_exc()

    def analyze_execution_plan_detailed(self, plan_tree: str) -> str:
        try:
            if not plan_tree or plan_tree.strip() == "":
                return "No MySQL execution plan available for analysis."

            analysis_prompt = f"""
            Analyze the following MySQL 8.0 execution plan in detail:

            **IMPORTANT**: This is a MySQL execution plan. Focus on MySQL-specific operations, terminology, and optimization opportunities.

            MySQL Execution Plan:
            {plan_tree}

            Please provide comprehensive MySQL-specific analysis including:

            1. **MySQL Operation Breakdown**:
               - Identify each MySQL operation in the plan tree
               - Explain the MySQL execution flow from inner to outer operations
               - Calculate total cost and time estimates using MySQL metrics

            2. **MySQL Performance Bottlenecks**:
               - Identify the most expensive MySQL operations
               - Find MySQL operations with high actual vs estimated times
               - Look for inefficient MySQL join strategies
               - Identify MySQL full table scans and missing indexes

            3. **MySQL Resource Usage Analysis**:
               - MySQL memory usage (sort operations, joins)
               - MySQL CPU intensive operations
               - MySQL I/O patterns and disk access

            4. **MySQL Row Processing Efficiency**:
               - Compare rows examined vs rows returned in MySQL context
               - Identify MySQL operations with poor selectivity
               - Find unnecessary MySQL data processing

            5. **MySQL Join Analysis**:
               - Analyze MySQL join algorithms (nested loop, hash, etc.)
               - Evaluate MySQL join order efficiency
               - Check for proper MySQL index usage in joins

            6. **MySQL Sort Operations**:
               - Identify expensive MySQL sort operations
               - Check for MySQL filesort usage
               - Analyze MySQL sort buffer requirements

            Focus on specific numeric values and provide MySQL-specific actionable insights.
            """

            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        except Exception as e:
            return f"Error analyzing MySQL execution plan: {str(e)}"

    def setup_graph(self):
        def prepare_analysis(state: AnalysisState) -> AnalysisState:
            try:
                efficiency = (
                    self.safe_division(
                        state["performance_metrics"]["rsent"],
                        state["performance_metrics"]["rexp"],
                    )
                    * 100
                )

                missing_metrics = []
                if state["performance_metrics"].get("rsent") is None:
                    missing_metrics.append("Rows Sent")
                if state["performance_metrics"].get("rexp") is None:
                    missing_metrics.append("Rows Examined")
                if state["performance_metrics"].get("lt") is None:
                    missing_metrics.append("Lock Time")

                data_availability_note = ""
                if missing_metrics:
                    data_availability_note = f"\n⚠️ **Note**: Some MySQL metrics are not available: {', '.join(missing_metrics)}. Analysis will focus on available MySQL data."

                formatted_schema = self.format_schema_info(state["schema_info"])

                qt_display = self.format_metric_display(
                    state["performance_metrics"]["qt"]
                )
                lt_display = self.format_metric_display(
                    state["performance_metrics"]["lt"]
                )
                rsent_display = self.format_metric_display(
                    state["performance_metrics"]["rsent"]
                )
                rexp_display = self.format_metric_display(
                    state["performance_metrics"]["rexp"]
                )

                analysis_context = f"""
                ## MySQL SQL Performance Analysis Request
                **IMPORTANT**: This is a MySQL database query analysis. All recommendations must be MySQL-compatible.
                {data_availability_note}

                ### MySQL Query to Analyze:
                ```sql
                {state['query']}
                ```

                ### MySQL Performance Metrics:
                - **Execution Time**: {qt_display} seconds
                - **Lock Time**: {lt_display} seconds  
                - **Rows Returned**: {rsent_display}
                - **Rows Examined**: {rexp_display}
                - **Efficiency**: {efficiency:.2f}% (calculated from available data)
                - **Timestamp**: {state['performance_metrics']['timestamp']}

                ### MySQL Execution Plan:
                ```
                {state['plan_tree']}
                ```

                {formatted_schema}

                ### MySQL Analysis Requirements:
                1. **MySQL Schema-Aware Analysis**: Utilize the provided MySQL schema information for accurate table structure analysis
                2. Analyze the MySQL execution plan tree structure (if available)
                3. Identify MySQL performance bottlenecks from available metrics
                4. Search for similar MySQL optimization examples
                5. Provide detailed MySQL technical analysis with schema context
                6. Focus on MySQL-specific optimization opportunities
                7. Consider MySQL table relationships and constraints from schema
                8. Handle missing MySQL metric data gracefully
                9. **Use proper Markdown formatting throughout the MySQL analysis**

                **Key MySQL Focus Areas:**
                - Leverage MySQL schema information for index recommendations
                - Consider MySQL table relationships for join optimization
                - Use MySQL schema constraints for query rewrite opportunities
                - MySQL execution time: {qt_display} seconds
                - MySQL efficiency: {efficiency:.2f}% (based on available row count data)
                """

                verification_context = f"""
                ## Original MySQL Query Data for Verification:
                **IMPORTANT**: This is MySQL database query verification. Ensure all technical checks are MySQL-specific.

                **MySQL Query:**
                ```sql
                {state['query']}
                ```

                **MySQL Performance Metrics:**
                - Execution Time: {qt_display} seconds
                - Lock Time: {lt_display} seconds
                - Rows Returned: {rsent_display}
                - Rows Examined: {rexp_display}
                - Efficiency: {efficiency:.2f}%
                - Timestamp: {state['performance_metrics']['timestamp']}

                **MySQL Execution Plan:**
                ```
                {state['plan_tree']}
                ```

                **MySQL Schema Information:**
                {formatted_schema}


                **TASK**: Verify the following MySQL performance analysis report against this original data and create a corrected, verified MySQL-compatible version.

                **MYSQL PERFORMANCE REPORT TO VERIFY:**
                """

                return {
                    **state,
                    "efficiency": efficiency,
                    "missing_metrics": missing_metrics,
                    "data_availability_note": data_availability_note,
                    "formatted_schema": formatted_schema,
                    "analysis_context": analysis_context,
                    "verification_context": verification_context,
                }

            except Exception as e:
                print(f"Error in prepare_analysis: {str(e)}")
                return {**state, "error": f"Preparation failed: {str(e)}"}

        def generate_report(state: AnalysisState) -> AnalysisState:
            try:
                if not self.report_agent_executor:
                    raise Exception("Report agent executor not initialized")

                report_result = self.report_agent_executor.invoke(
                    {"input": state["analysis_context"]}
                )

                performance_report = report_result["output"]

                return {**state, "performance_report": performance_report}

            except Exception as e:
                print(f"Error in generate_report: {str(e)}")
                return {**state, "error": f"Report generation failed: {str(e)}"}

        def generate_verified_report(state: AnalysisState) -> AnalysisState:
            try:
                if not self.verification_agent_executor:
                    raise Exception("Verification agent executor not initialized")

                if not state.get("performance_report"):
                    raise Exception("No performance report available for verification")

                verification_input = (
                    f"{state['verification_context']}\n\n{state['performance_report']}"
                )

                verification_result = self.verification_agent_executor.invoke(
                    {"input": verification_input}
                )

                verified_report = verification_result["output"]

                fact_check_results = self._extract_verification_metrics(verified_report)

                return {
                    **state,
                    "verified_report": verified_report,
                    "fact_check_results": fact_check_results,
                }

            except Exception as e:
                print(f"Error in generate_verified_report: {str(e)}")
                return {**state, "error": f"Verification failed: {str(e)}"}

        def generate_rewrite(state: AnalysisState) -> AnalysisState:
            try:
                if not self.rewrite_agent_executor:
                    raise Exception("Rewrite agent executor not initialized")

                qt_display = self.format_metric_display(
                    state["performance_metrics"]["qt"]
                )

                report_for_rewrite = state.get("verified_report") or state.get(
                    "performance_report", "Performance report not available"
                )

                rewrite_context = f"""
                {state['analysis_context']}


                ### MySQL Rewrite Requirements:
                **IMPORTANT**: Generate MySQL-compatible query rewrites only. All syntax must be valid for MySQL.

                1. **MySQL Schema-Based Optimization**: Use MySQL schema information for accurate table structure optimization
                2. Focus on optimizing the MySQL query execution (current time: {qt_display} seconds)
                3. Improve MySQL efficiency where possible (current: {state['efficiency']:.2f}%)
                4. Optimize MySQL subquery operations and UNION clauses
                5. Reduce MySQL table scan overhead using schema-aware indexing
                6. Provide multiple MySQL optimization approaches based on schema constraints
                7. Include necessary MySQL index recommendations using actual table structure
                8. Consider MySQL optimization even when some metrics are unavailable
                9. Leverage MySQL table relationships and foreign keys from schema
                10. **Use proper Markdown formatting with MySQL SQL code blocks**

                **MySQL Schema Context for Optimization:**
                {state['formatted_schema']}

                Generate MySQL-optimized query versions with detailed explanations and MySQL schema-aware improvements.
                """

                rewrite_result = self.rewrite_agent_executor.invoke(
                    {"input": rewrite_context}
                )

                query_rewrite = rewrite_result["output"]

                return {**state, "query_rewrite": query_rewrite}

            except Exception as e:
                print(f"Error in generate_rewrite: {str(e)}")
                return {**state, "error": f"Rewrite generation failed: {str(e)}"}

        def generate_summary(state: AnalysisState) -> AnalysisState:
            try:
                report_for_summary = state.get("verified_report") or state.get(
                    "performance_report"
                )

                if not report_for_summary or not state.get("query_rewrite"):
                    raise Exception(
                        "Performance report or query rewrite not available for summarization"
                    )

                summary = self.summary_agent.create_summary(
                    report_for_summary, state["query_rewrite"]
                )

                return {**state, "summary_report": summary}

            except Exception as e:
                print(f"Error in generate_summary: {str(e)}")
                return {**state, "error": f"Summary generation failed: {str(e)}"}

        def finalize_results(state: AnalysisState) -> AnalysisState:
            try:
                data_quality = "partial" if state["missing_metrics"] else "complete"
                schema_available = bool(
                    state["schema_info"]
                    and state["schema_info"] != "No schema available"
                )

                return {
                    **state,
                    "data_quality": data_quality,
                    "schema_available": schema_available,
                }

            except Exception as e:
                print(f"Error in finalize_results: {str(e)}")
                return {**state, "error": f"Result finalization failed: {str(e)}"}

        workflow = StateGraph(AnalysisState)

        workflow.add_node("prepare", prepare_analysis)
        workflow.add_node("report", generate_report)
        workflow.add_node("verify", generate_verified_report)
        workflow.add_node("rewrite", generate_rewrite)
        workflow.add_node("summary", generate_summary)
        workflow.add_node("finalize", finalize_results)

        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "report")
        workflow.add_edge("report", "verify")
        workflow.add_edge("verify", "rewrite")
        workflow.add_edge("rewrite", "summary")
        workflow.add_edge("summary", "finalize")
        workflow.add_edge("finalize", END)

        self.graph = workflow.compile()

    def _extract_verification_metrics(self, verified_report: str) -> dict:
        try:
            reliability_score = 8
            score_match = re.search(
                r"Reliability Score.*?(\d+(?:\.\d+)?)", verified_report, re.IGNORECASE
            )
            if score_match:
                try:
                    reliability_score = float(score_match.group(1))
                except ValueError:
                    reliability_score = 8

            confidence_level = "High"
            confidence_match = re.search(
                r"Confidence Level.*?(High|Medium|Low)", verified_report, re.IGNORECASE
            )
            if confidence_match:
                confidence_level = confidence_match.group(1).title()

            corrections_made = []
            corrections_match = re.search(
                r"Corrections Made.*?:(.*?)(?=\*\*|$)",
                verified_report,
                re.IGNORECASE | re.DOTALL,
            )
            if corrections_match:
                corrections_text = corrections_match.group(1).strip()
                if corrections_text and "none" not in corrections_text.lower():
                    corrections_made = [corrections_text]

            return {
                "reliability_score": reliability_score,
                "confidence_level": confidence_level,
                "verified_claims": 0,
                "total_claims": 0,
                "corrections_made": corrections_made,
            }

        except Exception as e:
            return {
                "reliability_score": 7,
                "confidence_level": "Medium",
                "verified_claims": 0,
                "total_claims": 0,
                "corrections_made": [],
            }

    def generate_comprehensive_analysis(
        self, json_input: dict, schema_info: dict = None
    ):
        try:
            metadata = json_input.get("query_metadata", {})
            explain_analyze = json_input.get("explain_analyze", {})

            query = metadata.get("query", "No query available")
            plan_tree = explain_analyze.get("plan_tree", "No execution plan available")

            db_schema = json_input.get("db_schema", {})
            if schema_info is None:
                schema_info = db_schema.get("schema_info", "No schema available")

            performance_metrics = {
                "qt": self.safe_numeric_value(metadata.get("qt")),
                "lt": self.safe_numeric_value(metadata.get("lt")),
                "rsent": self.safe_numeric_value(metadata.get("rsent")),
                "rexp": self.safe_numeric_value(metadata.get("rexp")),
                "timestamp": metadata.get("timestamp", "Not specified"),
            }

            initial_state = AnalysisState(
                query=query,
                plan_tree=plan_tree,
                performance_metrics=performance_metrics,
                schema_info=schema_info,
                efficiency=0.0,
                missing_metrics=[],
                data_availability_note="",
                formatted_schema="",
                analysis_context="",
                performance_report=None,
                query_rewrite=None,
                summary_report=None,
                verified_report=None,
                error=None,
                data_quality="unknown",
                schema_available=False,
                hallucination_report=None,
                fact_check_results=None,
                verification_context=None,
            )

            final_state = self.graph.invoke(initial_state)

            primary_performance_report = final_state.get(
                "verified_report"
            ) or final_state.get("performance_report")

            results = {
                "performance_report": primary_performance_report,  # display
                "original_report": final_state.get("performance_report"),
                "verified_report": final_state.get("verified_report"),
                "summary_report": final_state.get("summary_report"),  # display
                "query_rewrite": final_state.get("query_rewrite"),  # display
                "metrics": final_state.get("performance_metrics"),
                "efficiency": final_state.get("efficiency"),
                "missing_metrics": final_state.get("missing_metrics"),
                "data_quality": final_state.get("data_quality"),
                "schema_available": final_state.get("schema_available"),
                "hallucination_report": final_state.get("hallucination_report"),
                "fact_check_results": final_state.get("fact_check_results"),
                "error": final_state.get("error"),
            }

            return results

        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            import traceback

            traceback.print_exc()

            return {
                "error": f"Analysis failed: {str(e)}",
                "performance_report": None,
                "original_report": None,
                "verified_report": None,
                "summary_report": None,
                "query_rewrite": None,
                "metrics": None,
                "efficiency": 0,
                "schema_available": False,
                "hallucination_report": None,
                "fact_check_results": None,
            }


def run_ai_report_generation(session_id: str):
    input_key = f"analysis_inputs:{session_id}"
    report_key = f"report:{session_id}"

    try:
        input_data_json = redis_client.get(input_key)
        if not input_data_json:
            raise ValueError(f"Session id ({session_id}) not found in Redis.")

        analyzer = SQLQueryAnalyzer()
        result = analyzer.generate_comprehensive_analysis(
            json_input=json.loads(input_data_json)
        )

        report = {
            "status": "complete",
            "result": result,
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
        print(f"[{session_id}] Report saved to redis successfully")

    except Exception as e:
        print(f"[{session_id}] Background task failed: {e}")
        report = {
            "status": "failed",
            "result": str(e),
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
