import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

Configuración de Llaves y Modelo (Costo mínimo)
os.environ["OPENAI_API_KEY"] = "LLAVE
"
llm = ChatOpenAI(model="gpt-4o-mini")

Inicializamos la herramienta de búsqueda gratuita
search_tool = DuckDuckGoSearchRun()

Definición de Agentes ---

scraper_agent = Agent(
    role='Scraper de Noticias de Nearshoring',
    goal='Buscar en internet e identificar 2 nuevas empresas manufactureras estableciéndose en México.',
    backstory='Eres un investigador experto. Usas el buscador para encontrar noticias recientes sobre inversión extranjera en México.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool], # ¡Aquí le damos acceso a internet!
    llm=llm
)

enricher_agent = Agent(
    role='Enriquecedor de Datos Corporativos',
    goal='Buscar en internet información sobre los ingresos estimados y sector de las empresas identificadas.',
    backstory='Especialista en inteligencia de negocios. Sabes buscar el tamaño de las empresas basándote en noticias públicas.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool], # También necesita internet para investigar a la empresa
    llm=llm
)

analyst_agent = Agent(
    role='Analista de Inversiones PE',
    goal='Filtrar empresas y redactar un correo de contacto.',
    backstory='Asociado de un fondo de Private Equity. Eres directo y escribes correos persuasivos a CEOs.',
    verbose=True,
    allow_delegation=False,
    llm=llm # Este agente no necesita internet, solo analiza lo que los otros le dan
)

# Definición de Tareas ---

task_scraping = Task(
    description="Usa la herramienta de búsqueda para encontrar noticias de este año sobre 'empresas abriendo plantas en México nearshoring'. Extrae una lista de 2 nombres de empresas.",
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

# Ejecución del Proceso ---

sourcing_crew = Crew(
    agents=[scraper_agent, enricher_agent, analyst_agent],
    tasks=[task_scraping, task_enrichment, task_analysis],
    process=Process.sequential
)

# Esto iniciará el proceso. Verás en la consola cómo "piensa" cada agente.
result = sourcing_crew.kickoff()

print("\n\n########################")
print("## RESULTADO FINAL ##")
print("########################\n")
print(result)
