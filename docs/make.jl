using Documenter, ImplicitDomainQuadrature

makedocs(;
    modules=[ImplicitDomainQuadrature],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ArjunNarayanan/ImplicitDomainQuadrature/blob/{commit}{path}#L{line}",
    sitename="ImplicitDomainQuadrature.jl",
    authors="Arjun Narayanan",
    assets=String[],
)

deploydocs(;
    repo="github.com/ArjunNarayanan/ImplicitDomainQuadrature.git",
)
