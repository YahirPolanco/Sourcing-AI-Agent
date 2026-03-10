import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool # <--- Importamos la base oficial y más estable
from duckduckgo_search import DDGS

# 1. Configuración de Llaves y Modelo
os.environ["OPENAI_API_KEY"] = "Tu llave Open-Ai"
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Creamos la herramienta como una "Clase".
class MiBuscador(BaseTool):
    name: str = "Buscador de Internet"
    description: str = "Útil para buscar noticias, empresas e información financiera en internet."

    def _run(self, query: str) -> str:
        resultados = DDGS().text(query, max_results=3)
        return str(resultados)


search_tool = MiBuscador()

# Definición de Agentes ---

scraper_agent = Agent(
    role='Scraper de Noticias de Nearshoring',
    goal='Buscar en internet e identificar 2 nuevas empresas manufactureras estableciéndose en México.',
    backstory='Eres un investigador experto. Usas el buscador para encontrar noticias recientes sobre inversión extranjera en México.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool], # Usamos la herramienta de clase
    llm=llm
)

enricher_agent = Agent(
    role='Enriquecedor de Datos Corporativos',
    goal='Buscar en internet información sobre los ingresos estimados y sector de las empresas identificadas.',
    backstory='Especialista en inteligencia de negocios. Sabes buscar el tamaño de las empresas basándote en noticias públicas.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool], 
    llm=llm
)

analyst_agent = Agent(
    role='Analista de Inversiones PE',
    goal='Filtrar empresas y redactar un correo de contacto.',
    backstory='Asociado de un fondo de Private Equity. Eres directo y escribes correos persuasivos a CEOs.',
    verbose=True,
    allow_delegation=False,
    llm=llm 
)

# Definición de Tareas ---

task_scraping = Task(
    description="Usa el Buscador de Internet para encontrar noticias de este año sobre 'empresas abriendo plantas en México nearshoring'. Extrae una lista de 2 nombres de empresas.",
    agent=scraper_agent,
    expected_output="Una lista con los nombres de 2 empresas y un breve resumen de lo que hacen."
)

task_enrichment = Task(
    description="Toma la lista de empresas. Para cada una, busca en internet su facturación o tamaño de empleados estimado. Si no encuentras el dato exacto, da un estimado basado en la noticia de su inversión.",
    agent=enricher_agent,
    expected_output="Un reporte de las 2 empresas incluyendo tamaño, empleados e ingresos estimados."
)

task_analysis = Task(
    description="Toma el reporte. Finge que nuestro criterio es buscar empresas de manufactura. Redacta un correo de 3 líneas en español invitando al CEO a una llamada exploratoria con nuestro fondo de inversión.",
    agent=analyst_agent,
    expected_output="El borrador final de los correos electrónicos."
)

#Ejecución del Proceso ---

sourcing_crew = Crew(
    agents=[scraper_agent, enricher_agent, analyst_agent],
    tasks=[task_scraping, task_enrichment, task_analysis],
    process=Process.sequential
)

result = sourcing_crew.kickoff()

print("\n\n########################")
print("## RESULTADO FINAL ##")
print("########################\n")
print(result)