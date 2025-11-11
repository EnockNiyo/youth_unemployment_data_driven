"""
Real Data Sources Integration Module
Fetches youth unemployment data from multiple international sources
"""

import requests
import pandas as pd
import streamlit as st
from typing import Optional, Dict, List
import json
from datetime import datetime

class RealDataFetcher:
    """Fetch real-time unemployment data from various sources"""

    def __init__(self):
        self.world_bank_base_url = "https://api.worldbank.org/v2"
        self.ilo_base_url = "https://www.ilo.org/ilostat-files/WEB_bulk_download"

    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def fetch_world_bank_data(_self, indicator: str = "SL.UEM.1524.ZS",
                               country: str = "RW") -> Optional[pd.DataFrame]:
        """
        Fetch World Bank data for Rwanda
        SL.UEM.1524.ZS = Unemployment, youth total (% of total labor force ages 15-24)

        Args:
            indicator: World Bank indicator code
            country: Country code (RW for Rwanda)

        Returns:
            DataFrame with unemployment data or None
        """
        try:
            url = f"{_self.world_bank_base_url}/country/{country}/indicator/{indicator}"
            params = {
                "format": "json",
                "per_page": 100,
                "date": "2010:2024"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if len(data) > 1 and data[1]:
                records = []
                for item in data[1]:
                    if item['value'] is not None:
                        records.append({
                            'year': int(item['date']),
                            'country': item['country']['value'],
                            'indicator': item['indicator']['value'],
                            'value': float(item['value']),
                            'source': 'World Bank'
                        })

                df = pd.DataFrame(records)
                return df.sort_values('year', ascending=False)

            return None

        except requests.exceptions.RequestException as e:
            st.warning(f"World Bank API error: {e}")
            return None
        except Exception as e:
            st.error(f"Error processing World Bank data: {e}")
            return None

    @st.cache_data(ttl=86400)
    def fetch_multiple_indicators(_self, country: str = "RW") -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple unemployment-related indicators

        Returns:
            Dictionary of DataFrames with different indicators
        """
        indicators = {
            "youth_unemployment": "SL.UEM.1524.ZS",  # Youth unemployment (15-24)
            "youth_unemployment_male": "SL.UEM.1524.MA.ZS",  # Male youth unemployment
            "youth_unemployment_female": "SL.UEM.1524.FE.ZS",  # Female youth unemployment
            "unemployment_total": "SL.UEM.TOTL.ZS",  # Total unemployment
            "labor_force_participation": "SL.TLF.CACT.ZS",  # Labor force participation
            "gdp_growth": "NY.GDP.MKTP.KD.ZG"  # GDP growth (annual %)
        }

        results = {}
        for name, indicator in indicators.items():
            df = _self.fetch_world_bank_data(indicator, country)
            if df is not None:
                results[name] = df

        return results

    @st.cache_data(ttl=86400)
    def create_summary_statistics(_self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create summary statistics from multiple data sources

        Args:
            data_dict: Dictionary of indicator DataFrames

        Returns:
            Summary DataFrame
        """
        summary_data = []

        for name, df in data_dict.items():
            if not df.empty:
                latest = df.iloc[0]
                summary_data.append({
                    'indicator': name.replace('_', ' ').title(),
                    'latest_year': latest['year'],
                    'latest_value': round(latest['value'], 2),
                    'source': latest['source']
                })

        return pd.DataFrame(summary_data)

    def get_nisr_data(self, file_path: str = "nisr_dataset.csv") -> Optional[pd.DataFrame]:
        """
        Load and process NISR dataset (already real data!)

        Args:
            file_path: Path to NISR dataset

        Returns:
            Processed NISR DataFrame
        """
        try:
            df = pd.read_csv(file_path, encoding='latin1')

            # Add metadata
            df['data_source'] = 'NISR Rwanda'
            df['data_type'] = 'Survey Data'

            return df
        except Exception as e:
            st.error(f"Error loading NISR data: {e}")
            return None

    @st.cache_data(ttl=86400)
    def fetch_african_dev_bank_data(_self) -> Optional[pd.DataFrame]:
        """
        Placeholder for African Development Bank data
        Note: AfDB doesn't have a public API, but data can be downloaded manually

        Returns:
            DataFrame or None
        """
        # This would be populated with manually downloaded data
        # or scraped data (with permission)
        st.info("African Development Bank data requires manual download from: https://dataportal.opendataforafrica.org/")
        return None

    def combine_all_sources(self, nisr_path: str = "nisr_dataset.csv") -> Dict[str, any]:
        """
        Combine data from all available sources

        Args:
            nisr_path: Path to NISR dataset

        Returns:
            Dictionary with all data sources
        """
        combined_data = {}

        # World Bank data
        wb_data = self.fetch_multiple_indicators()
        combined_data['world_bank'] = wb_data

        # NISR data
        nisr_data = self.get_nisr_data(nisr_path)
        combined_data['nisr'] = nisr_data

        # Summary statistics
        if wb_data:
            summary = self.create_summary_statistics(wb_data)
            combined_data['summary'] = summary

        return combined_data


class DataQualityChecker:
    """Check data quality and provide metadata"""

    @staticmethod
    def check_data_freshness(df: pd.DataFrame, date_column: str = 'year') -> Dict:
        """
        Check how fresh the data is

        Args:
            df: DataFrame to check
            date_column: Column containing date/year

        Returns:
            Dictionary with freshness metrics
        """
        if df is None or df.empty:
            return {"status": "No data", "latest_year": None}

        latest_year = df[date_column].max()
        current_year = datetime.now().year
        years_old = current_year - latest_year

        status = "Fresh" if years_old <= 1 else "Outdated" if years_old > 3 else "Recent"

        return {
            "status": status,
            "latest_year": latest_year,
            "years_old": years_old,
            "records": len(df)
        }

    @staticmethod
    def get_data_coverage(df: pd.DataFrame) -> Dict:
        """
        Get data coverage statistics

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with coverage metrics
        """
        if df is None or df.empty:
            return {"coverage": "No data"}

        total_cols = len(df.columns)
        total_rows = len(df)
        missing_pct = (df.isnull().sum().sum() / (total_cols * total_rows)) * 100

        return {
            "total_columns": total_cols,
            "total_rows": total_rows,
            "completeness": f"{100 - missing_pct:.1f}%",
            "missing_percentage": f"{missing_pct:.1f}%"
        }


# Example usage function
def display_real_data_dashboard():
    """Display real data in Streamlit dashboard"""
    st.header("📊 Real-Time Data Sources")

    fetcher = RealDataFetcher()
    checker = DataQualityChecker()

    # Fetch all data
    with st.spinner("Fetching real-time data from international sources..."):
        all_data = fetcher.combine_all_sources()

    # Display World Bank data
    if 'world_bank' in all_data and all_data['world_bank']:
        st.subheader("🌍 World Bank Data")

        # Show summary
        if 'summary' in all_data:
            st.dataframe(all_data['summary'], use_container_width=True)

        # Show detailed indicators
        with st.expander("View Detailed World Bank Indicators"):
            for name, df in all_data['world_bank'].items():
                if df is not None and not df.empty:
                    st.write(f"**{name.replace('_', ' ').title()}**")
                    st.dataframe(df.head(10))

                    # Data quality check
                    freshness = checker.check_data_freshness(df)
                    st.info(f"Data Status: {freshness['status']} (Latest: {freshness['latest_year']})")

    # Display NISR data info
    if 'nisr' in all_data and all_data['nisr'] is not None:
        st.subheader("🇷🇼 NISR Rwanda Survey Data")

        nisr_df = all_data['nisr']
        coverage = checker.get_data_coverage(nisr_df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", coverage['total_rows'])
        with col2:
            st.metric("Data Completeness", coverage['completeness'])
        with col3:
            st.metric("Columns", coverage['total_columns'])

    return all_data


if __name__ == "__main__":
    # Test the module
    display_real_data_dashboard()
