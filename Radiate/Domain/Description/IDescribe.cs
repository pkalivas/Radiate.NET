namespace Radiate.Domain.Description;

interface IDescribe
{
    string Describe<T>(T data);
}